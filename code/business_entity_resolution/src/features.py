"""Pairwise feature engineering for candidate (S1, S2/S3) pairs.

Features are computed ONLY for candidate pairs (never the full Cartesian
product). Everything is derived from the supplied data (plus the documented
auditable normalization rules - no external information).

Runtime note: RapidFuzz is a C library, but the per-pair loop is still the
slowest stage of the baseline. ``light_features`` trades a few metrics for
speed; lowering ``block_top_k``/``max_candidates_per_s1`` cuts volume directly.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------

BASE_COLUMNS = [
    "rank", "score", "best_gap", "cand_count",
]

NAME_COLUMNS = [
    "name_ratio", "name_token_set", "name_token_sort",
    "name_jw", "name_len_ratio", "name_tok_jaccard",
]

ADDR_COLUMNS = [
    "addr_ratio", "addr_token_set", "addr_jaccard",
    "addr_char3_jaccard", "addr_len_ratio",
]

NUMERIC_COLUMNS = [
    "num_overlap_frac", "num_conflict", "postal_equal", "postal_any",
]

OTHER_COLUMNS = ["country_equal", "source_is_s3"]

FEATURE_COLUMNS = BASE_COLUMNS + NAME_COLUMNS + ADDR_COLUMNS + NUMERIC_COLUMNS + OTHER_COLUMNS

# Cheaper variant (fewer RapidFuzz calls per pair).
LIGHT_COLUMNS = [
    "rank", "score", "best_gap", "cand_count",
    "name_ratio", "name_token_set", "name_len_ratio",
    "addr_ratio", "addr_jaccard",
    "num_overlap_frac", "postal_equal", "country_equal", "source_is_s3",
]


# ---------------------------------------------------------------------------
# Similarity helpers (lazy import of rapidfuzz keeps import cheap)
# ---------------------------------------------------------------------------

def _rf():
    from rapidfuzz import fuzz
    return fuzz


def _safe_ratio(a, b):
    if not a or not b:
        return 0.0
    return _rf().ratio(a, b) / 100.0


def _safe_token_set(a, b):
    if not a or not b:
        return 0.0
    return _rf().token_set_ratio(a, b) / 100.0


def _safe_token_sort(a, b):
    if not a or not b:
        return 0.0
    return _rf().token_sort_ratio(a, b) / 100.0


def _safe_jw(a, b):
    if not a or not b:
        return 0.0
    from rapidfuzz.distance import JaroWinkler
    return JaroWinkler.normalized_similarity(a, b)


def _tok_jaccard(a, b):
    sa, sb = a.split(), b.split()
    if not sa or not sb:
        return 0.0
    A, B = set(sa), set(sb)
    return len(A & B) / len(A | B)


def _char3_jaccard(a, b):
    if not a or not b:
        return 0.0
    A = {a[i:i + 3] for i in range(max(0, len(a) - 2))}
    B = {b[i:i + 3] for i in range(max(0, len(b) - 2))}
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def _num_overlap(a, b):
    sa, sb = a.split(), b.split()
    if not sa or not sb:
        return 0.0
    return len(set(sa) & set(sb)) / max(len(set(sa)), len(set(sb)))


def _num_conflict(a, b):
    sa, sb = set(a.split()), set(b.split())
    return 1.0 if (sa and sb and not (sa & sb)) else 0.0


def _len_ratio(a, b):
    if not a or not b:
        return 0.0
    la, lb = len(a), len(b)
    return min(la, lb) / max(la, lb)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_features(cands, s1_tab, s23_tab, cfg, verbose=True):
    """Append feature columns to ``cands`` and return ``(df, X, names)``.

    ``s1_tab`` / ``s23_tab``: entity tables indexed by ``entity_id`` with the
    normalized columns needed (norm_name, norm_addr, num_tok, postal_tok,
    country, source).
    """
    cols = LIGHT_COLUMNS if cfg.light_features else FEATURE_COLUMNS

    df = cands.copy()

    s1 = s1_tab.set_index("entity_id")
    s23 = s23_tab.set_index("entity_id")

    get_s1 = lambda col: df["s1"].astype(str).map(s1[col])
    get_s23 = lambda col: df["s23"].astype(str).map(s23[col])

    # --- numeric / postal / country -------------------------------------
    n1 = get_s1("num_tok").fillna("")
    n2 = get_s23("num_tok").fillna("")
    p1 = get_s1("postal_tok").fillna("")
    p2 = get_s23("postal_tok").fillna("")

    if "num_overlap_frac" in cols:
        df["num_overlap_frac"] = [ _num_overlap(a, b) for a, b in zip(n1, n2)]
    if "num_conflict" in cols:
        df["num_conflict"] = [_num_conflict(a, b) for a, b in zip(n1, n2)]
    if "postal_equal" in cols:
        df["postal_equal"] = [1.0 if a and a == b else 0.0 for a, b in zip(p1, p2)]
    if "postal_any" in cols:
        df["postal_any"] = [1.0 if bool(a) or bool(b) else 0.0 for a, b in zip(p1, p2)]
    if "country_equal" in cols:
        df["country_equal"] = (get_s1("country").astype(str) == get_s23("country").astype(str)).astype(np.float32)
    if "source_is_s3" in cols:
        df["source_is_s3"] = df["s23"].astype(str).str.startswith("S3-").astype(np.float32)

    # --- contextual (already present on the candidates frame) ------------
    if "cand_count" in cols:
        df["cand_count"] = df.groupby("s1", observed=True)["s23"].transform("count").astype(np.float32)
    if "score" in cols:
        df["score"] = df["score"].astype(np.float32)
    if "best_gap" in cols:
        best = df.groupby("s1", observed=True)["score"].transform("max")
        df["best_gap"] = (best - df["score"]).astype(np.float32)

    # --- name / address (RapidFuzz pass) ----------------------------------
    nn1 = get_s1("norm_name").fillna("")
    aa1 = get_s1("norm_addr").fillna("")
    nn2 = get_s23("norm_name").fillna("")
    aa2 = get_s23("norm_addr").fillna("")

    if "name_ratio" in cols:
        df["name_ratio"] = [_safe_ratio(a, b) for a, b in zip(nn1, nn2)]
    if "name_token_set" in cols:
        df["name_token_set"] = [_safe_token_set(a, b) for a, b in zip(nn1, nn2)]
    if "name_token_sort" in cols:
        df["name_token_sort"] = [_safe_token_sort(a, b) for a, b in zip(nn1, nn2)]
    if "name_jw" in cols:
        df["name_jw"] = [_safe_jw(a, b) for a, b in zip(nn1, nn2)]
    if "name_len_ratio" in cols:
        df["name_len_ratio"] = [_len_ratio(a, b) for a, b in zip(nn1, nn2)]
    if "name_tok_jaccard" in cols:
        df["name_tok_jaccard"] = [_tok_jaccard(a, b) for a, b in zip(nn1, nn2)]

    if "addr_ratio" in cols:
        df["addr_ratio"] = [_safe_ratio(a, b) for a, b in zip(aa1, aa2)]
    if "addr_token_set" in cols:
        df["addr_token_set"] = [_safe_token_set(a, b) for a, b in zip(aa1, aa2)]
    if "addr_jaccard" in cols:
        df["addr_jaccard"] = [_tok_jaccard(a, b) for a, b in zip(aa1, aa2)]
    if "addr_char3_jaccard" in cols:
        df["addr_char3_jaccard"] = [_char3_jaccard(a, b) for a, b in zip(aa1, aa2)]
    if "addr_len_ratio" in cols:
        df["addr_len_ratio"] = [_len_ratio(a, b) for a, b in zip(aa1, aa2)]

    X = df[cols].to_numpy(dtype=np.float32)
    return df, X, cols


def add_labels(df, gt_map):
    """Add a binary label column: 1 iff (s1, s23) is a ground-truth pair.

    ``gt_map``: dict S1 -> set(children). Singletons have empty sets so their
    candidate pairs become hard negatives by construction.
    """
    import numpy as _np

    lookup = df["s1"].astype(str).map(lambda s: gt_map.get(s, set()))
    hits = [child in found for found, child in zip(lookup, df["s23"].astype(str))]
    df["y"] = _np.where(hits, 1, 0).astype(_np.int8)
    return df


def downsample_negatives(df, ratio: float, seed: int):
    """Keep ``ratio`` negatives per positive (for LightGBM training)."""
    import pandas as pd

    if ratio <= 0:
        return df[df["y"] == 1]
    pos = df[df["y"] == 1]
    n_pos = len(pos)
    neg = df[df["y"] == 0]
    if n_pos and len(neg) > n_pos * ratio:
        neg = neg.sample(n=int(n_pos * ratio), random_state=seed)
    return pd.concat([pos, neg], ignore_index=True).sample(frac=1.0, random_state=seed)