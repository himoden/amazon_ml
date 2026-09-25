"""Decision layer: per-S1 set selection + global one-parent resolution.

The competition predicts a *set* of S2/S3 ids per S1. We:

1. sort each S1's candidates by model probability (descending);
2. accept the first while p >= ``first_threshold`` and each further candidate
   while p >= ``add_threshold`` (add-ons require more evidence --- precision
   is weighted ~2x over recall);
3. cap at ``max_pred_per_s1``;
4. optionally apply the global one-parent constraint (A1 confirmed by EDA:
   every S2/S3 child has exactly one S1 parent) --- keep the highest-probability
   parent and drop the child from the others.

Thresholds are tuned on the validation fold against macro F0.5, never by
intuition.
"""

from __future__ import annotations

import numpy as np


def sorted_per_s1(df, prob_col="prob"):
    """Split candidate-probability rows into per-S1 lists sorted desc by prob."""
    groups = {}
    for s1, prob in zip(df["s1"], df[prob_col]):
        groups.setdefault(s1, []).append(prob)
    for s1 in groups:
        groups[s1].sort(reverse=True)
    return groups


def greedy_decision(df, first_threshold, add_threshold,
                    max_pred=None, prob_col="prob"):
    """Apply the greedy top-prefix rule per S1.

    Returns ``dict S1 -> list of matched S2/S3 ids``.
    """
    out = {}
    groups = sorted_per_s1(df, prob_col)
    for s1, probs in groups.items():
        chosen = []
        if probs and probs[0] >= first_threshold:
            chosen.append(probs[0])
            for p in probs[1:]:
                if p >= add_threshold:
                    chosen.append(p)
                else:
                    break
        if max_pred:
            chosen = chosen[:max_pred]
        out[s1] = chosen
    return out


def decision_ids_from_frame(df, first_threshold, add_threshold,
                            max_pred=None, prob_col="prob"):
    """Directly produce the accepted (s1, s23) rows for the greedy rule.

    Faster than going through probabilities only (avoids a second pass).
    Returns a DataFrame [s1, s23, prob, rank].
    """
    import pandas as pd

    df = df.sort_values([prob_col], ascending=False)
    taken = []
    by_s1 = {}
    for s1, s23, prob in zip(df["s1"], df["s23"], df[prob_col]):
        lst = by_s1.setdefault(s1, [])
        if not lst and prob >= first_threshold:
            lst.append((prob, s23))
        elif lst and prob >= add_threshold and len(lst) < (max_pred or np.inf):
            lst.append((prob, s23))
    rows = [(s1, s23, prob) for s1, lst in by_s1.items() for prob, s23 in lst]
    out = pd.DataFrame(rows, columns=["s1", "s23", prob_col])
    if not out.empty:
        out["s1"] = out["s1"].astype("category")
        out[prob_col] = out[prob_col].astype(np.float32)
    return out


def sets_from_frame(accepted):
    """dict S1 -> set of matched ids (for the scorer)."""
    out = {}
    for s1, s23 in zip(accepted["s1"], accepted["s23"]):
        out.setdefault(s1, set()).add(s23)
    return out


# ---------------------------------------------------------------------------
# Global one-parent resolution
# ---------------------------------------------------------------------------

def apply_one_parent(accepted):
    """Enforce A1: each S2/S3 child belongs to exactly one S1 parent.

    Among all S1s that accepted a given child, keep the highest-probability
    assignment. Returns the adjusted (s1, s23) frame.
    """
    if len(accepted) == 0:
        return accepted
    best = accepted.loc[accepted.groupby("s23")["prob"].idxmax()]
    # a tie (identical probability) keeps the first row; groupby.idxmax handles it
    return best.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------

def tune_thresholds(df, gt_map, cfg, scoring=None):
    """Grid-search (first_threshold, add_threshold) on macro F0.5.

    ``df`` : candidate rows for the validation-fold S1s with a ``prob`` column.
    ``gt_map`` : ground-truth dict for those S1s.
    Returns ``(best_params, best_score, table)``.
    """
    import itertools

    from . import scoring as sc

    best = None
    best_score = -1.0
    table = []
    first_grid = cfg.threshold_grid_first
    add_grid = cfg.threshold_grid_add
    for ft, at in itertools.product(first_grid, add_grid):
        if at < ft:
            continue
        accepted = decision_ids_from_frame(
            df, ft, at, max_pred=cfg.max_pred_per_s1)
        if cfg.use_one_parent:
            accepted = apply_one_parent(accepted)
        preds = sets_from_frame(accepted)
        true = {s1: gt_map.get(s1, set()) for s1 in set(df["s1"])}
        r = sc.metrics_report(true, preds, empty_non_singleton_score=cfg.empty_non_singleton_score)
        row = (ft, at, r["macro_f05"], r["non_singleton_recall"],
               r["singleton_accuracy"], r["avg_predictions_per_s1"])
        table.append(row)
        if r["macro_f05"] > best_score:
            best_score = r["macro_f05"]
            best = (ft, at)
    return best, best_score, table


def print_tuning_table(table, top=12):
    import pandas as pd

    df = pd.DataFrame(table, columns=["first_t", "add_t", "macro_f05",
                                      "non_sing_recall", "sing_acc", "avg_pred"])
    df = df.sort_values("macro_f05", ascending=False)
    print(df.head(top).to_string(index=False))