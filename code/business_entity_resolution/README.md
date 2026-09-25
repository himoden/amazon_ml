# Business Entity Resolution - baseline pipeline (Amazon ML Challenge 2026)

CPU-only baseline: lexical blocking -> similarity features -> LightGBM ->
per-S1 set decision -> global one-parent resolution -> TSV outputs.

See `notebooks/amazon_ml_challenge_baseline.ipynb` for end-to-end execution on
Kaggle / Google Colab, and `reports/notebook_execution_guide.md` for setup.

## Layout

```
code/business_entity_resolution/
├── requirements.txt
├── README.md
└── src/
    ├── config.py        # all knobs; paths derived from DATA_ROOT
    ├── data.py          # TSV loading; entity-table caching (parquet)
    ├── normalize.py     # Unicode-safe normalization + auditable Devanagari trans.
    ├── blocking.py      # TF-IDF char-ngram top-k retrieval + exact-key pass
    ├── features.py      # pairwise similarity features (RapidFuzz)
    ├── scoring.py       # exact macro F0.5 scorer (+ output consistency checks)
    ├── model.py         # LightGBM wrapper
    ├── decision.py      # set selection + one-parent + threshold tuning
    └── pipeline.py      # phase orchestrator (idempotent stages)
```

## How to run

Do **not** run the heavy pipeline on a laptop. Execute the notebook in a
Kaggle/Colab runtime:

1. Add the dataset to the runtime (`/kaggle/input/...` or upload to Colab).
2. Set `DATA_ROOT` in the notebook config cell (auto-detected on Kaggle/Colab).
3. Run cells top-to-bottom; cache folders let you resume after disconnects.

Optional standalone checks (light, safe anywhere):

```bash
python -m pytest tests/ -q          # pure-python scorer/normalize tests
python tests/test_sanity.py         # tiny synthetic run (no dataset access)
```

## Compliance notes (see also PS.md / PATH.md)

- No external business/registry/geocoding/API data anywhere in this code.
- `country` is an open set; nothing hard-codes `{US, India}`.
- `scoring.empty_non_singleton_score` is the **provisional** A3 convention
  (0.0) and is labelled as provisional in every report.
- The Devanagari transliteration table in `normalize.py` is a manual,
  auditable rule used ONLY for blocking keys; the Unicode view is preserved.
- Model: LightGBM (MIT). Dense/reranker/LLM stages are intentionally deferred.

## Reproducibility

- `SEED = cfg.seed` (default 42) fixed in config + model params.
- `requirements.txt` pinned; versions printed by the notebook after install.
- All heavy intermediates are parquet-cached under `cache/`.