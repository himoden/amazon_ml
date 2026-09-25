"""Data loading, caching, ground-truth parsing.

Memory / scale design
---------------------
The full dataset is ~26.4M records (~2.5 GB raw). This module:

* loads ONE source file at a time (each file is up to ~5M rows);
* normalizes it immediately and writes a compact parquet cache to ``cache_dir``;
* never keeps every DataFrame alive at once (del / gc are used after caching).

Callers therefore work on the cached entity tables, loaded per country or per
source (train/test), keeping peak RAM bounded.
"""

from __future__ import annotations

import os

_COLS = ["entity_id", "business_name", "business_address", "country"]

FILES = {
    ("train", "s1"): "train_source1.tsv",
    ("train", "s2"): "train_source2.tsv",
    ("train", "s3"): "train_source3.tsv",
    ("test", "s1"): "test_source1.tsv",
    ("test", "s2"): "test_source2.tsv",
    ("test", "s3"): "test_source3.tsv",
}
GT_FILE = "train_ground_truth.tsv"

_PARQUET = {
    ("train", "s1"): "entities_train_s1.parquet",
    ("train", "s2"): "entities_train_s2.parquet",
    ("train", "s3"): "entities_train_s3.parquet",
    ("test", "s1"): "entities_test_s1.parquet",
    ("test", "s2"): "entities_test_s2.parquet",
    ("test", "s3"): "entities_test_s3.parquet",
}


def load_source_tsv(path: str, nrows: int | None = None):
    """Load one source TSV. Returns a *raw* DataFrame (str dtype).

    Reading is done inside the strict ``sep="\\t"`` contract from the problem
    statement. We do NOT convert to categories here (that happens after cache
    load); entity ids stay as plain strings.
    """
    import pandas as pd

    return pd.read_csv(path, sep="\t", dtype=str, usecols=_COLS, nrows=nrows)


def _read_nrows(cfg, source: str, nrows) -> int:
    """Apply the smoke-test row limits when enabled."""
    if not cfg.smoke:
        return None
    return cfg.smoke_s1_limit if source == "s1" else cfg.smoke_pool_limit


def build_entity_table(raw, source: str, cfg, keep_raw: bool = True):
    """Normalize a raw source DataFrame into the compact entity table.

    Columns:
      entity_id, source, country, raw_name, raw_addr, norm_name, norm_addr,
      latn_name, latn_addr, search (latn_name + ' ' + latn_addr),
      core_key, num_tok, postal_tok.
    """
    from . import normalize as N

    out = raw[["entity_id"]].copy()
    out["source"] = source
    out["country"] = raw["country"].astype("category")

    if keep_raw:
        out["raw_name"] = raw["business_name"]
        out["raw_addr"] = raw["business_address"]
    else:
        out["raw_name"] = raw["business_name"]
        out["raw_addr"] = raw["business_address"]

    # normalized views (Unicode-preserving)
    out["norm_name"] = raw["business_name"].map(N.normalize_text)
    out["norm_addr"] = raw["business_address"].map(N.normalize_text)

    # latin-fold views for retrieval/blocking keys only
    out["latn_name"] = raw["business_name"].map(N.latin_fold)
    out["latn_addr"] = raw["business_address"].map(N.latin_fold)
    out["search"] = out["latn_name"] + " " + out["latn_addr"]

    out["core_key"] = out["norm_name"].map(N.core_key)
    out["num_tok"] = raw["business_address"].map(N.numeric_token_string)
    out["postal_tok"] = raw["business_address"].map(N.postal_token_string)

    del raw
    return out


def _cache_path(cfg, split: str, source: str) -> str:
    return os.path.join(cfg.paths().cache_dir, _PARQUET[(split, source)])


def load_or_build_entities(cfg, split: str, source: str):
    """Load cached entities or (re)build them. Returns the entity DataFrame."""
    import pandas as pd

    path = _cache_path(cfg, split, source)
    if cfg.cache_entities and os.path.isfile(path):
        return pd.read_parquet(path)

    src = FILES[(split, source)]
    raw_path = os.path.join(cfg.paths().dataset_dir, "train" if split == "train" else "test", src)
    nrows = _read_nrows(cfg, source, None)
    raw = load_source_tsv(raw_path, nrows=nrows)
    tab = build_entity_table(raw, source, cfg, keep_raw=cfg.keep_raw)
    if cfg.smoke:
        # keep only one country if the whole-pool smoke test would be huge
        if source in ("s2", "s3"):
            tab = tab.groupby("country", dropna=False).head(cfg.smoke_pool_limit)
    if cfg.cache_entities:
        tab.to_parquet(path, index=False)
    return tab


def load_train(cfg):
    return {s: load_or_build_entities(cfg, "train", s) for s in ("s1", "s2", "s3")}


def load_test(cfg):
    return {s: load_or_build_entities(cfg, "test", s) for s in ("s1", "s2", "s3")}


def remove_entities(tabs: dict) -> None:
    """Free memory explicitly (notebook keeps only what the current stage needs)."""
    import gc

    for k in list(tabs):
        del tabs[k]
    gc.collect()


def load_ground_truth(cfg):
    """Parse train_ground_truth.tsv -> dict S1 -> set{children}. NaN/empty = {}."""
    import pandas as pd

    path = os.path.join(cfg.paths().dataset_dir, "train", GT_FILE)
    gt = pd.read_csv(path, sep="\t", dtype=str)
    maps = {}
    for s1, m in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        if pd.isna(m) or not str(m).strip():
            maps[s1] = set()
            continue

        parts = [p.strip() for p in str(m).split(",") if p.strip()]
        maps[s1] = set(parts)
    return maps


def load_ground_truth_frame(cfg):
    import pandas as pd

    path = os.path.join(cfg.paths().dataset_dir, "train", GT_FILE)
    return pd.read_csv(path, sep="\t", dtype=str)


def resolve_ids_by_prefix(entities: dict, prefix: str):
    """Set of all entity ids in the given source prefix across loaded tables."""
    ids = set()
    for source, tab in entities.items():
        if tab["entity_id"].astype(str).str.startswith(prefix).any():
            ids.update(tab["entity_id"].tolist())
    return ids