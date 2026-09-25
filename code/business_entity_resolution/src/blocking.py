"""Candidate generation / blocking.

Scale-aware design
------------------
The full comparison space is S1 x (S2 u S3) = ~1.7M x ~10M (test) - the
notebook NEVER materialises that product. Instead we:

1. build a per-country TF-IDF (char n-gram) index over the S2/S3 *pool*;
2. top-k retrieve per S1 in memory-bounded chunks (``query_chunk``);
3. union with cheap exact-key passes;
4. dedupe, rank (score desc) and cap each S1 at ``max_candidates_per_s1``.

Country blocking is ON by default because EDA (A2) proved every ground-truth
pair shares the same country label. France is handled identically: it is just
another country key (never hard-coded).
"""

from __future__ import annotations

import os

import numpy as np


class TfidfRetriever:
    """Character n-gram TF-IDF + cosine top-k retrieval over a pool.

    Internally stores a float32 CSR matrix of the pool. Query rows are
    processed in chunks of ``query_chunk``; each chunk is materialised as a
    small dense (chunk x n_pool) scores matrix only long enough to select the
    top-k (this bounds peak memory).
    """

    def __init__(self, ngram_range=(3, 5), max_features=300_000, min_df=2):
        self.ngram_range = tuple(ngram_range)
        self.max_features = int(max_features)
        self.min_df = int(min_df)
        self.vectorizer = None
        self.pool_ids = None
        self.pool_mat = None
        self.id_to_row = None

    # ------------------------------------------------------------------
    def _l2_normalize_rows(self, mat, dtype=np.float32):
        norms = np.sqrt(np.asarray(mat.multiply(mat).sum(axis=1)).ravel())
        norms[norms == 0.0] = 1.0
        mat = mat.multiply(1.0 / norms[:, None]).tocsr()
        return mat.astype(dtype) if dtype else mat

    def build(self, corpus, ids):
        """Fit the vectorizer on the pool corpus and store the pool matrix."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            min_df=self.min_df,
            sublinear_tf=True,
        )
        self.pool_mat = self._l2_normalize_rows(
            self.vectorizer.fit_transform(corpus), np.float32
        )
        self.pool_ids = np.asarray(ids, dtype=object)
        self.id_to_row = {pid: i for i, pid in enumerate(self.pool_ids)}
        return self

    def transform(self, corpus):
        if self.vectorizer is None:
            raise RuntimeError("TfidfRetriever.build() must be called first")
        return self._l2_normalize_rows(
            self.vectorizer.transform(corpus), np.float32
        )

    def retrieve_topk_matrix(self, query_mat, k, chunk=64):
        """Top-k (pool_id, score) per query row from an already-transformed matrix.

        Returns lists aligned to query rows: ``out[i] = [(pool_id, score), ...]``
        sorted by descending score.
        """
        k = max(1, int(k))
        n_q = query_mat.shape[0]
        out = [[] for _ in range(n_q)]
        pool = self.pool_mat
        for start in range(0, n_q, chunk):
            end = min(start + chunk, n_q)
            sub = query_mat[start:end]
            dense = (sub @ pool.T).toarray()  # (chunk, n_pool) float32
            for r in range(dense.shape[0]):
                row = dense[r]
                nz = len(row)
                if nz == 0:
                    continue
                take = min(k, nz)
                idx = np.argpartition(-row, kth=take - 1)[:take]
                scores = row[idx]
                order = np.argsort(-scores, kind="stable")
                idx = idx[order]
                scores = scores[order]
                out[start + r] = [
                    (self.pool_ids[i], float(scores[j]))
                    for j, i in enumerate(idx)
                    if np.isfinite(scores[j])
                ]
        return out

    def retrieve_topk_corpus(self, corpus, k, chunk=64):
        """Top-k retrieval over raw (string) query corpus."""
        return self.retrieve_topk_matrix(self.transform(corpus), k, chunk=chunk)


# ---------------------------------------------------------------------------
# Exact-key blocking
# ---------------------------------------------------------------------------

def build_corekey_index(pool):
    """dict: core_key -> list of pool row indices."""
    idx = {}
    core = pool["core_key"].to_numpy(dtype=object)
    for i, key in enumerate(core):
        if key:
            idx.setdefault(key, []).append(i)
    return idx


def exact_corekey_candidates(s1_tab, pool, core_idx):
    """Pairs sharing an identical core key (S1 rows x pool rows, score 1.0).

    Returns (s1_ids, pool_ids, scores) numpy arrays (may be empty).
    """
    rows = []
    core = s1_tab["core_key"].to_numpy(dtype=object)
    pool_ids = pool["entity_id"].to_numpy(dtype=object)
    s1_ids = s1_tab["entity_id"].to_numpy(dtype=object)
    for i, key in enumerate(core):
        if not key:
            continue
        hits = core_idx.get(key)
        if not hits:
            continue
        for r in hits:
            rows.append((s1_ids[i], pool_ids[r], 1.0))
    if not rows:
        return (np.empty(0, dtype=object), np.empty(0, dtype=object),
                np.empty(0, dtype=np.float32))
    arr = np.array(rows, dtype=object)
    return (arr[:, 0], arr[:, 1], np.ones(len(rows), dtype=np.float32))


# ---------------------------------------------------------------------------
# Full candidate generation
# ---------------------------------------------------------------------------

def generate_candidates_for_country(s1_tab, pool, cfg, retriever=None):
    """Create the ranked candidate set for one country.

    * exact core-key pass (score 1.0)
    * TF-IDF top-k retrieval (name+address latin-fold text)
    * dedupe, rank, cap to ``max_candidates_per_s1``

    Returns a DataFrame columns [s1, s23, score, rank].
    """
    import pandas as pd

    if len(s1_tab) == 0 or len(pool) == 0:
        return pd.DataFrame(columns=["s1", "s23", "score", "rank"])

    frames = []

    # 1) exact keys ---------------------------------------------------------
    if cfg.use_exact_keys:
        core_idx = build_corekey_index(pool)
        s1a, p23a, sc_a = exact_corekey_candidates(s1_tab, pool, core_idx)
        if len(s1a):
            frames.append(pd.DataFrame(
                {"s1": s1a, "s23": p23a, "score": sc_a}))

    # 2) retrieval ----------------------------------------------------------
    if retriever is None:
        retriever = TfidfRetriever(
            ngram_range=cfg.ngram_range,
            max_features=cfg.max_features,
            min_df=cfg.min_df,
        ).build(pool["search"].tolist(), pool["entity_id"].tolist())

    results = retriever.retrieve_topk_corpus(
        s1_tab["search"].tolist(), cfg.block_top_k, chunk=cfg.query_chunk
    )
    if results:
        flat = []
        s1_ids = s1_tab["entity_id"].tolist()
        for q_idx, hits in enumerate(results):
            sid = s1_ids[q_idx]
            for pid, score in hits:
                if cfg.percentile_floor and score < cfg.percentile_floor:
                    continue
                flat.append((sid, pid, score))
        if flat:
            frames.append(pd.DataFrame(
                flat, columns=["s1", "s23", "score"]))

    if not frames:
        return pd.DataFrame(columns=["s1", "s23", "score", "rank"])

    cand = pd.concat(frames, ignore_index=True)
    del frames

    # dedupe: keep the highest score per (s1, s23)
    cand = cand.groupby(["s1", "s23"], as_index=False)["score"].max()
    cand["score"] = cand["score"].clip(lower=0.0, upper=1.0)

    # rank within S1 by descending score, then cap
    cand["rank"] = cand.groupby("s1")["score"].rank(
        method="first", ascending=False).astype(np.int32) - 1
    cand = cand[cand["rank"] < cfg.max_candidates_per_s1].reset_index(drop=True)
    cand = cand.sort_values(["s1", "rank"]).reset_index(drop=True)
    cand["s1"] = cand["s1"].astype("category")
    cand["score"] = cand["score"].astype(np.float32)
    cand["rank"] = cand["rank"].astype(np.int16)
    return cand


def generate_candidates_all_countries(s1_tab, pool, cfg):
    """Run blocking per country and stack the frames (country block = ON)."""
    import pandas as pd

    frames = []
    for country in sorted(pool["country"].cat.categories):
        s1_c = s1_tab[s1_tab["country"] == country]
        pool_c = pool[pool["country"] == country]
        if len(s1_c) == 0 or len(pool_c) == 0:
            continue
        frames.append(generate_candidates_for_country(s1_c, pool_c, cfg))
    if not frames:
        return pd.DataFrame(columns=["s1", "s23", "score", "rank"])
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------

def blocking_recall(cands, gt_map, s1_ids=None):
    """Fraction of ground-truth children present in the candidate set.

    ``s1_ids`` restricts the evaluation to a subset of S1 entities (e.g. the
    validation fold). Children of singleton S1s are ignored (trivially found).
    """
    import pandas as pd

    if s1_ids is not None:
        s1_ids = set(s1_ids)
        required = [(s1, ch) for s1, ch in gt_map.items()
                    if s1 in s1_ids and ch]
    else:
        required = [(s1, ch) for s1, ch in gt_map.items() if ch]

    if not required:
        return 1.0, 0, 0, 0

    # candidate children per restricted S1
    if s1_ids is not None:
        cands = cands[cands["s1"].isin(s1_ids)]
    pool_by_s1 = cands.groupby("s1", observed=True)["s23"].apply(set).to_dict()

    total = found = 0
    for s1, children in required:
        hits = children & pool_by_s1.get(s1, set())
        total += len(children)
        found += len(hits)
    recall = found / total if total else 1.0
    return recall, found, total, len(required)


def candidate_stats(cands, n_s1, n_pool):
    """(avg candidates per S1, reduction ratio, total pairs)."""
    total = len(cands)
    uniq_s1 = cands["s1"].nunique() if len(cands) else n_s1
    avg = total / uniq_s1 if uniq_s1 else 0.0
    full = n_s1 * n_pool
    reduced = 1.0 - (total / full) if full else 1.0
    return avg, reduced, total


def save_candidates(cands, path):
    cands.to_parquet(path, index=False)


def load_candidates(path):
    import pandas as pd

    return pd.read_parquet(path)


def cache_candidates_if(path, cands, cache: bool):
    if cache and path and not os.path.isfile(path):
        save_candidates(cands, path)