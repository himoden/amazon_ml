"""Pipeline orchestration.

Each phase is a thin, idempotent function the notebook calls in-order. Heavy
intermediates (normalized entity tables, candidate sets) are parquet-cached so
a runtime disconnect can be resumed by simply re-running the notebook.

Phases
------
1. prepare            : resolve paths, mkdirs, env summary
2. build_entities     : normalize + cache per source file (train/test)
3. build_gt           : ground-truth maps
4. group_split        : S1-grouped train/validation split
5. candidates_train   : blocking over the train pool         (train + valid folds)
6. features_train     : pairwise features + labels            (fold-dependent)
7. train_model        : LightGBM on the train fold
8. validate_fold      : predict valid fold, tune thresholds, one-parent, report
9. final_model        : retrain on full train data (optional)
10. candidates_test   : blocking over the test pool
11. infer_test        : features + inference on test candidates
12. decide_test       : set selection + one-parent -> matching sets
13. write_outputs     : matching_results.tsv + candidate_pairs.tsv
14. run_validator     : run the official validator
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from . import blocking as B
from . import features as F
from . import scoring as S
from . import decision as DEC
from . import model as M


def _t(msg, start):
    print(C.timer_report(time.time() - start, msg))


def load_or_build(path, builder, force=False):
    """Load a parquet cache or run ``builder()`` and persist it.

    Used for candidate frames so a runtime disconnect can be resumed without
    re-running blocking. ``force=True`` ignores any existing file.
    """
    if path and not force and os.path.isfile(path):
        return pd.read_parquet(path)
    out = builder()
    if path:
        out.to_parquet(path, index=False)
    return out


# ---------------------------------------------------------------------------
# Phase 1 - prepare
# ---------------------------------------------------------------------------

def prepare(cfg: C.Config):
    cfg.ensure_dirs()
    print("DATA_ROOT :", cfg.data_root)
    print(cfg.env_summary())
    return cfg.paths()


def setup(cfg: C.Config, override_data_root: str | None = None) -> C.Config:
    root, env = C.detect_environment(override_data_root)
    if override_data_root is not None:
        cfg.data_root = override_data_root
    elif not cfg.data_root:
        cfg.data_root = root
        print(f"[setup] detected environment: {env}, DATA_ROOT={root}")
    prepare(cfg)
    return cfg


# ---------------------------------------------------------------------------
# Phase 2 - normalized entity tables (cached)
# ---------------------------------------------------------------------------

def build_entities(cfg: C.Config, splits=("train", "test")) -> dict:
    """Normalize + cache each source file. Returns {split: {source: tab}}.

    Loads one file at a time; peak memory is bounded by the largest source
    (test S3 ~ 5.1M rows). Pass ``splits=("train",)`` first if RAM is tight.
    """
    t0 = time.time()
    tabs = {}
    for split in splits:
        tabs[split] = {}
        for source in ("s1", "s2", "s3"):
            tab = D.load_or_build_entities(cfg, split, source)
            tabs[split][source] = tab
            print(f"  [{split}/{source}] rows={len(tab)} cached="
                  f"{D._cache_path(cfg, split, source)}")
    _t("build_entities", t0)
    return tabs


def entity_summary(cfg: C.Config) -> pd.DataFrame:
    """Rows x countries / null counts for all sources."""
    rows = []
    for split in ("train", "test"):
        for source in ("s1", "s2", "s3"):
            tab = D.load_or_build_entities(cfg, split, source)
            raw_name = tab["raw_name"]
            raw_addr = tab["raw_addr"]
            entry = {
                "split": split,
                "source": source,
                "rows": len(tab),
                "empty_name": int(raw_name.isna().sum() + (raw_name == "").sum()),
                "empty_addr": int(raw_addr.isna().sum() + (raw_addr == "").sum()),
            }
            for c in sorted(tab["country"].cat.categories):
                entry[str(c)] = int((tab["country"] == c).sum())
            rows.append(entry)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Phase 3 - ground truth
# ---------------------------------------------------------------------------

def build_gt(cfg: C.Config) -> dict:
    t0 = time.time()
    gt = D.load_ground_truth(cfg)
    n_sing = sum(1 for v in gt.values() if not v)
    n_pairs = sum(len(v) for v in gt.values())
    print(f"[gt] {len(gt)} S1 entities, {n_sing} singletons "
          f"({n_sing / len(gt):.3%}), {n_pairs} positive pairs")
    _t("build_gt", t0)
    return gt


# ---------------------------------------------------------------------------
# Phase 4 - S1-grouped split
# ---------------------------------------------------------------------------

def split_ids(s1_ids, seed: int = 42, fraction: float = 0.05):
    """Deterministic grouped shuffle split returning (train, valid)."""
    seq = list(s1_ids)
    rng = np.random.RandomState(seed)
    rng.shuffle(seq)
    n_valid = max(1, int(len(seq) * fraction))
    return seq[n_valid:], seq[:n_valid]


def group_split(cfg: C.Config, s1_ids):
    if cfg.full_validation:
        return list(s1_ids), list(s1_ids)
    train_ids, valid_ids = split_ids(s1_ids, cfg.seed, cfg.validation_fraction)
    print(f"[split] train S1={len(train_ids)}, valid S1={len(valid_ids)}")
    return train_ids, valid_ids


# ---------------------------------------------------------------------------
# Phase 5+6 - candidates + features for the training data
# ---------------------------------------------------------------------------

def pool_frame(train_tabs: dict, s1_countries=None) -> pd.DataFrame:
    """S2+S3 combined table (for the given split)."""
    pool = pd.concat([train_tabs["s2"], train_tabs["s3"]], ignore_index=True)
    if s1_countries is not None:
        pool["country"] = pool["country"].cat.set_categories(
            list(s1_countries) + [
                c for c in pool["country"].cat.categories
                if c not in s1_countries])
    return pool


def candidates_train(cfg: C.Config, train_tabs, gt_map, train_ids, valid_ids):
    """Blocking over the full train S2/S3 pool (country-block inside).

    Returns (cands_train, cands_valid, stats_dict).
    """
    t0 = time.time()
    s1 = train_tabs["s1"]
    pool = pool_frame(train_tabs, s1["country"].cat.categories.tolist())
    cands_all = B.generate_candidates_all_countries(s1, pool, cfg)

    import pandas as pd

    cands_all["fold"] = cands_all["s1"].map(
        {i: "valid" for i in valid_ids}).fillna("train").astype("category")
    cands_train = cands_all[cands_all["fold"] == "train"].drop(columns="fold")
    cands_valid = cands_all[cands_all["fold"] == "valid"].drop(columns="fold")

    recall_v, found, total, n_req = B.blocking_recall(
        cands_valid, gt_map, s1_ids=valid_ids)
    avg_v, reduced_v, total_pairs = B.candidate_stats(
        cands_valid, n_s1=len(valid_ids), n_pool=len(pool))
    print(f"[blocking-train] valid-fold recall={recall_v:.4f} "
          f"({found}/{total} children, {n_req} non-singletons), "
          f"avg_cands/S1={avg_v:.2f}, reduction={reduced_v:.4%}")
    _t("candidates_train", t0)
    stats = {"valid_recall": recall_v, "avg_cands": avg_v, "reduction": reduced_v,
             "train_pairs": len(cands_train), "valid_pairs": len(cands_valid)}
    return cands_train, cands_valid, stats


def build_features(cfg, cands, train_tabs):
    """Pairwise features for a candidate frame (no labels here)."""
    t0 = time.time()
    df, X, names = F.build_features(
        cands, train_tabs["s1"], pool_frame(train_tabs), cfg)
    _t("features", t0)
    return df, X, names


def add_train_labels(cfg, df, gt_map):
    df = F.add_labels(df, gt_map)
    pos = int((df["y"] == 1).sum())
    print(f"[labels] candidate pairs={len(df)}, positives={pos}")
    return df


# ---------------------------------------------------------------------------
# Phase 7 - model training
# ---------------------------------------------------------------------------

def train_model(cfg, df_train, feature_names):
    """Train the baseline LightGBM on train-fold candidates.

    ``df_train`` must already have the feature columns plus ``y``.
    """
    df_m = df_train[list(feature_names) + ["y"]].copy()
    df_m = F.downsample_negatives(df_m, cfg.negative_sample_ratio, cfg.seed)
    X = df_m[list(feature_names)].to_numpy(dtype=np.float32)
    y = df_m["y"].to_numpy()

    # small intra-train holdout for early stopping
    rng = np.random.RandomState(cfg.seed)
    perm = rng.permutation(len(X))
    n_h = max(1, len(X) // 20)
    val_idx, tr_idx = perm[:n_h], perm[n_h:]

    model, best_it, _ = M.train_lgbm(
        X[tr_idx], y[tr_idx], X[val_idx], y[val_idx], cfg)
    if cfg.verbose:
        imp = M.feature_importance(model, list(feature_names), top=20)
        print(imp.to_string(index=False))
    return model


# ---------------------------------------------------------------------------
# Phase 8 - validation
# ---------------------------------------------------------------------------

def validate_fold(cfg, model, df_valid, X_valid, gt_map, valid_ids, feature_names):
    """Predict validation candidates, tune thresholds, one-parent, report."""
    probs = M.predict_proba(model, X_valid)
    dfv = df_valid[["s1", "s23"]].copy()
    dfv["prob"] = probs
    (best_first, best_add), best_score, table = DEC.tune_thresholds(
        dfv, gt_map, cfg)
    DEC.print_tuning_table(table)
    print(f"[decision] best (first={best_first}, add={best_add}) "
          f"-> macro F0.5 = {best_score:.4f}")
    # final report using the best thresholds
    accepted = DEC.decision_ids_from_frame(
        dfv, best_first, best_add, max_pred=cfg.max_pred_per_s1)
    if cfg.use_one_parent:
        accepted = DEC.apply_one_parent(accepted)
    preds = DEC.sets_from_frame(accepted)
    true = {s1: gt_map.get(s1, set()) for s1 in set(dfv["s1"])}
    country_of = _country_map(cfg)
    report = S.metrics_report(true, preds, entity_country=country_of,
                              empty_non_singleton_score=cfg.empty_non_singleton_score)
    print(S.format_report(report))
    return {"best_first": best_first, "best_add": best_add,
            "best_macro_f05": best_score, "report": report,
            "tuning_table": table}


def _country_map(cfg):
    """S1 -> country dict from the cached train S1 entities."""
    tab = D.load_or_build_entities(cfg, "train", "s1")
    return dict(zip(tab["entity_id"], tab["country"].astype(str)))


def predict_valid(cfg, model, df_valid, X_valid):
    probs = M.predict_proba(model, X_valid)
    dfv = df_valid[["s1", "s23"]].copy()
    dfv["prob"] = probs
    return dfv


# ---------------------------------------------------------------------------
# Phase 9 - final model (retrain on all train candidates)
# ---------------------------------------------------------------------------

def final_model(cfg, df_all, feature_names):
    """Retrain on ALL train S1 candidates using the tuned thresholds' data."""
    return train_model(cfg, df_all, feature_names)


# ---------------------------------------------------------------------------
# Phase 10-13 - test inference & outputs
# ---------------------------------------------------------------------------

def candidates_test(cfg, test_tabs):
    t0 = time.time()
    s1 = test_tabs["s1"]
    pool = pool_frame(test_tabs, s1["country"].cat.categories.tolist())
    cands = B.generate_candidates_all_countries(s1, pool, cfg)
    avg, reduced, total = B.candidate_stats(cands, len(s1), len(pool))
    print(f"[blocking-test] candidates={len(cands)}, avg/S1={avg:.2f}, "
          f"reduction={reduced:.4%}")
    _t("candidates_test", t0)
    return cands, pool


def infer_test(cfg, model, cands_test, test_tabs):
    df, X, names = F.build_features(cands_test, test_tabs["s1"], pool_frame(test_tabs), cfg)
    probs = M.predict_proba(model, X)
    df["prob"] = probs
    return df[["s1", "s23", "prob"]], names


def decide_test(cfg, df_test_probs, test_s1_ids, first_threshold=0.95, add_threshold=0.98):
    """Set selection + optional one-parent over the full test candidate frame."""
    accepted = DEC.decision_ids_from_frame(
        df_test_probs, first_threshold, add_threshold, max_pred=cfg.max_pred_per_s1)
    if cfg.use_one_parent:
        accepted = DEC.apply_one_parent(accepted)
    matches = DEC.sets_from_frame(accepted)
    for s1 in test_s1_ids:
        matches.setdefault(s1, set())
    print(f"[decision-test] S1 with >=1 predicted match: "
          f"{sum(1 for v in matches.values() if v)}")
    return matches


def write_outputs(cfg, matches, cands_test, test_tabs,
                  write_matches=True, write_candidates=True):
    """matching_results.tsv (matches) + candidate_pairs.tsv (full blocking set).

    candidate_pairs.tsv is intentionally the FULL candidate set fed to the
    matcher (matches subset: ``matches`` then always satisfy ``matches subseteq
    candidates``).

    ``write_matches`` / ``write_candidates`` allow the two files to be emitted
    separately (the notebook writes matching_results first, then candidates).
    """
    os.makedirs(cfg.paths().output_dir, exist_ok=True)
    all_s1 = test_tabs["s1"]["entity_id"].tolist()

    mpath = os.path.join(cfg.paths().output_dir, "matching_results.tsv")
    if write_matches:
        rows = [(s1, ",".join(sorted(matches.get(s1, set())))) for s1 in all_s1]
        matches_df = pd.DataFrame(rows, columns=["source1_entity_id", "matched_entity_ids"])
        matches_df.to_csv(mpath, sep="\t", index=False, encoding="utf-8")

    cpath = os.path.join(cfg.paths().output_dir, "candidate_pairs.tsv")
    if write_candidates:
        cand_rows = []
        grouped = cands_test.groupby("s1", observed=True)["s23"].apply(list).to_dict()
        for s1 in all_s1:
            cand_rows.append((s1, ",".join(sorted(set(grouped.get(s1, []))))))
        cand_df = pd.DataFrame(cand_rows, columns=["source1_entity_id", "candidate_entity_ids"])
        cand_df.to_csv(cpath, sep="\t", index=False, encoding="utf-8")

    print(f"wrote:\n  {mpath if write_matches else '(skipped)'}\n  {cpath if write_candidates else '(skipped)'}")
    return mpath, cpath


def run_validator(cfg, matching_path, candidate_path, check_ids=False,
                  verbose=True):
    """Run the official validator; returns (ok, captured_output)."""
    import subprocess
    import sys

    vpath = cfg.paths().validator
    test_dir = os.path.join(cfg.paths().dataset_dir, "test")
    cmd = [sys.executable, vpath, "--matching", matching_path,
           "--candidate", candidate_path, "--test-dir", test_dir]
    if check_ids:
        cmd.append("--check-ids")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout
    if proc.stderr:
        out += "\n" + proc.stderr
    if verbose:
        print(out)
    ok = proc.returncode == 0
    print("VALIDATOR: " + ("PASS" if ok else "FAIL"))
    return ok, out


def experiment_summary(cfg, metrics) -> str:
    lines = [
        "=" * 60,
        "AMAZON ML CHALLENGE 2026 - BASELINE EXPERIMENT SUMMARY",
        "=" * 60,
        f"seed                     : {cfg.seed}",
        f"block_top_k              : {cfg.block_top_k}",
        f"max_candidates_per_s1    : {cfg.max_candidates_per_s1}",
        f"ngram_range              : {cfg.ngram_range}",
        f"max_features             : {cfg.max_features}",
        f"validation_fraction      : {cfg.validation_fraction}",
        f"model                    : {cfg.model}",
        f"negative_sample_ratio    : {cfg.negative_sample_ratio}",
        f"one_parent               : {cfg.use_one_parent}",
        "A3 empty-non-singleton convention (provisional): "
        f"{cfg.empty_non_singleton_score}",
        f"macro F0.5 (valid)       : {metrics.get('macro_f05', float('nan')):.4f}",
        f"blocking recall (valid)  : {metrics.get('valid_recall', float('nan')):.4f}",
        f"avg candidates/S1 (valid): {metrics.get('avg_cands', float('nan')):.2f}",
        f"reduction ratio (valid)  : {metrics.get('reduction', float('nan')):.4%}",
        "=" * 60,
    ]
    return "\n".join(lines)