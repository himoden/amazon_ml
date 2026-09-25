"""Exact local F0.5 scorer for the competition.

Method
------
For each Source-1 entity ``s1``:

    precision = |T n P| / |P|
    recall    = |T n P| / |T|
    F0.5      = 1.25 * P * R / (0.25 * P + R)

then macro-average over all S1 entities in the evaluation set.

Singleton rule (documented in the official README):
    true = {}  and pred = {}  -> score 1.0
    true = {}  and pred != {} -> score 0.0

A3 (UNVERIFIED): an *empty prediction on a non-singleton* is not covered by
the README. We use the provisional convention ``empty_non_singleton_score =
0.0`` and clearly label it everywhere; it must be treated as a provisional
local choice, not an organizer rule. The audio worked example from the
README (P=2/3, R=2/2 -> F=0.714) is reproduced in the unit tests.
"""

from __future__ import annotations


def f05_score(precision: float, recall: float) -> float:
    if precision <= 0 and recall <= 0:
        return 0.0
    denom = 0.25 * precision + recall
    if denom <= 0:
        return 0.0
    return 1.25 * precision * recall / denom


def per_entity_score(true_set, pred_set, empty_non_singleton_score: float = 0.0) -> float:
    """Score for a single S1 entity.

    ``empty_non_singleton_score`` is the provisional A3 convention for
    ``true != {} and pred == {}``.
    """
    t = set(true_set)
    p = set(pred_set or [])
    if not t:
        return 1.0 if not p else 0.0
    if not p:
        return float(empty_non_singleton_score)
    tp = len(t & p)
    precision = tp / len(p)
    recall = tp / len(t)
    return f05_score(precision, recall)


def single_entity_metrics(true_set, pred_set):
    """Return (precision, recall) pair or None when undefined."""
    t, p = set(true_set), set(pred_set or [])
    if not t or not p:
        return None
    tp = len(t & p)
    return (tp / len(p), tp / len(t))


def metrics_report(true_map, pred_map, entity_country=None,
                   empty_non_singleton_score: float = 0.0):
    """macro-F0.5 report over S1 entities.

    Params
    ------
    true_map / pred_map : dict S1 -> set(children)
    entity_country      : optional dict S1 -> country for per-country report

    Returns a dict with overall and (when provided) per-country metrics.
    """
    ids = [s1 for s1 in pred_map if s1 in true_map]
    if not ids:
        return {"n_entities": 0, "n_singletons": 0, "macro_f05": 0.0}

    scores = []
    singleton_correct = singleton_total = 0
    non_sing_recovered = non_sing_total = 0
    empty_for_non_sing = non_sing_count = 0
    precision_sum = recall_sum = 0.0
    n_scored_pairwise = 0

    for s1 in ids:
        t, p = true_map[s1], pred_map[s1]
        s = per_entity_score(t, p, empty_non_singleton_score)
        scores.append(s)

        if not t:
            singleton_total += 1
            singleton_correct += int(not p)
        else:
            non_sing_total += len(t)
            non_sing_count += 1
            if not p:
                empty_for_non_sing += 1
            m = single_entity_metrics(t, p)
            if m is not None:
                precision_sum += m[0]
                recall_sum += m[1]
                n_scored_pairwise += 1
            non_sing_recovered += len(set(t) & set(p))

    report = {
        "n_entities": len(ids),
        "n_singletons": singleton_total,
        "n_non_singletons": non_sing_count,
        "macro_f05": 0.0,
        "mean_precision": (precision_sum / n_scored_pairwise) if n_scored_pairwise else float("nan"),
        "mean_recall": (recall_sum / n_scored_pairwise) if n_scored_pairwise else float("nan"),
        "singleton_accuracy": (singleton_correct / singleton_total) if singleton_total else float("nan"),
        "non_singleton_recall": (non_sing_recovered / non_sing_total) if non_sing_total else float("nan"),
        "non_singleton_empty_rate": (empty_for_non_sing / non_sing_count) if non_sing_count else float("nan"),
        "avg_predictions_per_s1": _avg(len(pred_map[s1]) for s1 in ids),
        "n_total_true_pairs": sum(len(v) for v in true_map.values()),
        "empty_non_singleton_score": empty_non_singleton_score,
    }
    report["macro_f05"] = float(sum(scores) / len(scores))

    if entity_country is not None:
        per_country = {}
        countries = {}
        for s1 in ids:
            countries.setdefault(entity_country.get(s1), []).append(s1)
        for ctry, cids in countries.items():
            sub_true = {s1: true_map[s1] for s1 in cids}
            sub_pred = {s1: pred_map[s1] for s1 in cids}
            cr = metrics_report(sub_true, sub_pred, None, empty_non_singleton_score)
            per_country[ctry] = {
                "macro_f05": cr["macro_f05"],
                "n_entities": cr["n_entities"],
                "singleton_accuracy": cr["singleton_accuracy"],
                "non_singleton_recall": cr["non_singleton_recall"],
                "empty_rate": cr["non_singleton_empty_rate"],
            }
        report["per_country"] = per_country

    return report


def _avg(values):
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def format_report(report) -> str:
    lines = [
        "F0.5 evaluation (macro over S1 entities):",
        f"  entities             : {report['n_entities']}",
        f"  singletons           : {report['n_singletons']}",
        f"  non-singletons       : {report['n_non_singletons']}",
        f"  macro F0.5           : {report['macro_f05']:.4f}   (A3 empty-pred convention = {report['empty_non_singleton_score']})",
        f"  mean precision       : {report['mean_precision']:.4f}",
        f"  mean recall          : {report['mean_recall']:.4f}",
        f"  singleton accuracy   : {report['singleton_accuracy']:.4f}",
        f"  non-singleton recall : {report['non_singleton_recall']:.4f}",
        f"  non-sing. empty rate : {report['non_singleton_empty_rate']:.4f}",
        f"  avg preds/S1         : {report['avg_predictions_per_s1']:.3f}",
    ]
    pc = report.get("per_country")
    if pc:
        lines.append("  per-country:")
        for c, r in pc.items():
            lines.append(
                f"    {c}: F0.5={r['macro_f05']:.4f} n={r['n_entities']} "
                f"sing_acc={r['singleton_accuracy']:.4f} "
                f"non_sing_recall={r['non_singleton_recall']:.4f} "
                f"empty={r['empty_rate']:.4f}"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Output-consistency checks (mirror the official validator's rules)
# ---------------------------------------------------------------------------

def validate_outputs(matches_map, candidates_map, required_ids,
                     valid_ids=None, verbose=True):
    """Return a list of error strings (empty list == OK).

    mirrors ``utils/validate_submission.py`` so failures are caught early.
    ``valid_ids`` = set of real test S2/S3 ids (optional, but recommended once,
    since building it costs a few GB of RAM on the full test set).
    """
    errors = []
    missing = set(required_ids) - set(matches_map)
    if missing:
        errors.append(f"{len(missing)} required S1 entities missing a row")
    extra = set(matches_map) - set(required_ids)
    if extra:
        errors.append(f"{len(extra)} rows use S1 ids not in the test set")
    for s1, preds in matches_map.items():
        preds = set(preds or [])
        if len(preds) != len(set(preds)):
            errors.append(f"{s1}: duplicate ids inside its match list")
        if any(p.startswith("S1-") for p in preds):
            errors.append(f"{s1}: self-matches (S1 ids) present")
        if any(not (p.startswith("S2-") or p.startswith("S3-")) for p in preds):
            errors.append(f"{s1}: id without S2-/S3- prefix")
        if valid_ids is not None and (preds - valid_ids):
            errors.append(f"{s1}: ids not in the test S2/S3 files")
    if candidates_map is not None:
        for s1, preds in matches_map.items():
            cands = set(candidates_map.get(s1, set()) or [])
            if set(preds or []) - cands:
                errors.append(f"{s1}: matched ids missing from candidates (pipeline bug)")
    return errors