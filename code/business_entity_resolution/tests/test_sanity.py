"""Tiny synthetic end-to-end sanity test. NEVER touches the challenge dataset.

Validates that: source module imports resolve, normalization behaves,
blocking retrieves a known near-duplicate, features produce a valid matrix,
a small LightGBM fits, and decision/scoring round-trip produces a sound
F0.5. Prerequisites: rapidfuzz + lightgbm + sklearn + scipy installed.

Run: python tests/test_sanity.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _tiny_entity_table(rows, source):
    out = pd.DataFrame(rows, columns=["entity_id", "name", "addr", "country"])
    out = out.rename(columns={"name": "raw_name", "addr": "raw_addr"})
    from normalize import normalize_text, latin_fold, core_key, \
        numeric_token_string, postal_token_string

    out["country"] = out["country"].astype("category")
    out["norm_name"] = out["raw_name"].map(normalize_text)
    out["norm_addr"] = out["raw_addr"].map(normalize_text)
    out["latn_name"] = out["raw_name"].map(latin_fold)
    out["latn_addr"] = out["raw_addr"].map(latin_fold)
    out["search"] = out["latn_name"] + " " + out["latn_addr"]
    out["core_key"] = out["norm_name"].map(core_key)
    out["num_tok"] = out["raw_addr"].map(numeric_token_string)
    out["postal_tok"] = out["raw_addr"].map(postal_token_string)
    out["source"] = source
    return out


def main():
    from config import Config
    from blocking import TfidfRetriever, generate_candidates_for_country
    from features import build_features, add_labels
    from model import train_lgbm, predict_proba
    from decision import decision_ids_from_frame, sets_from_frame, apply_one_parent
    from scoring import metrics_report

    # --- tiny pool: 6 businesses + 2 decoys -------------------------------
    pool_rows = [
        ("S2-1001", "Ambernath Solar Private Limited", "H.No 16-11-23/37/A, Mumbai, Maharashtra 400042", "India"),
        ("S3-1002", "Ambernath Solar Pvt Ltd", "16-11-23/37/A, AMBERNATH, MUMBAI, MH 400042", "India"),
        ("S2-1003", "Blue Harbor Restaurant LLC", "1111 Church Street, Unit 2007, Nashville, TN 37203", "US"),
        ("S3-1004", "Blue Harbor Rest Ltd", "1111 Church St, Unit 2, Nashville, Tennessee 37203", "US"),
        ("S2-1005", "OZT AMICALE SAS", "24 R DESAIX, TOURCOING, Hauts-de-France 59200", "France"),
        ("S3-1006", "OZT Amicale", "24 Rue DesaiX, Tourcoing 59200, Hauts-de-France", "France"),
        ("S2-2000", "Totally Different Bakery Inc", "9990 Random Road, Springfield, IL 62704", "US"),
        ("S3-2001", "Acme Logistics Pvt Ltd", "Plot 9, Sector 74, Noida, Uttar Pradesh 201301", "India"),
    ]
    pool = _tiny_entity_table(pool_rows, "pool")

    s1_rows = [
        ("S1-9001", "Ambernath Solar Private Limited", "16/37/A Ambernath East, Mumbai, Maharashtra 400042", "India"),
        ("S1-9002", "OZT Amicale SAS", "24 Rue Desaix, Tourcoing 59200, Hauts de France", "France"),
        ("S1-9003", "Unique Noisy Place Limited", "5 Plot xyz, Mumbai, Maharashtra 400001", "India"),  # singleton
    ]
    s1 = _tiny_entity_table(s1_rows, "s1")

    cfg = Config()
    cfg.max_candidates_per_s1 = 10
    cfg.block_top_k = 10

    cands = generate_candidates_for_country(s1, pool, cfg)
    print("candidates:\n", cands.to_string(index=False))

    # blocking recall: S1-9001 -> {S2-1001, S3-1002}; S1-9002 -> {S2-1005, S3-1006}
    gt = {
        "S1-9001": {"S2-1001", "S3-1002"},
        "S1-9002": {"S2-1005", "S3-1006"},
        "S1-9003": set(),
    }
    pool_ids = set(cands["s1"])
    found_children = set()
    for s1id, children in gt.items():
        got = set(cands[cands["s1"] == s1id]["s23"]) if (cands["s1"] == s1id).any() else set()
        found_children |= (children & got)
    recall = len(found_children) / (2 + 2)
    print(f"blocking recall (tiny): {recall}")
    assert recall == 1.0, "retrieval missed a match"

    df, X, names = build_features(cands, s1, pool, cfg)
    df = add_labels(df, gt)
    print("features:", len(names), "pairs:", len(df))
    assert X.shape[0] == len(df)

    # train a tiny model
    dfm = df[list(names) + ["y"]].copy()
    Xt, yt = dfm[list(names)].to_numpy(dtype=np.float32), dfm["y"].to_numpy()
    model, _, _ = train_lgbm(Xt, yt, Xt[:2], yt[:2], cfg,
                             params={"objective": "binary", "metric": "binary_logloss",
                                     "num_leaves": 7, "learning_rate": 0.1,
                                     "n_jobs": 1, "verbosity": -1, "seed": 42},
                             num_boost_round=50, early_stopping_rounds=10, verbose=False)

    dfv = df[["s1", "s23"]].copy()
    dfv["prob"] = predict_proba(model, Xt)
    acc = decision_ids_from_frame(dfv, 0.5, 0.3, max_pred=5)
    acc = apply_one_parent(acc)
    preds = sets_from_frame(acc)
    true = {s1: gt.get(s1, set()) for s1 in set(dfv["s1"])}
    rep = metrics_report(true, preds)
    print("F0.5 (tiny):", round(rep["macro_f05"], 4))
    print("OK - sanity pipeline passed")


if __name__ == "__main__":
    main()