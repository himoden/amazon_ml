# Notebook Execution Guide - Amazon ML Challenge 2026 (baseline)

Target: execute `notebooks/amazon_ml_challenge_baseline.ipynb` on **Kaggle** or
**Google Colab** (CPU is enough for the baseline). The laptop is only the dev
environment; never run the heavy cells locally.

Reference files:
- `PS.md`, `PATH.md` (progress rules / phase log)
- `DATASET_ANALYSIS.md`, `reports/eda.md` (design evidence)
- `code/business_entity_resolution/` (modules imported by the notebook)

---

## 1. Add the dataset to the platform

### Kaggle

1. In Kaggle, go to **Datasets** -> **New Dataset** (upper right).
2. Upload the `student_resource/` folder **as-is** (it already contains
   `dataset/` and `utils/`). Name it e.g. `mlchallenge2026-er`.
   > Upload the *.tsv files directly (do NOT zip them) so the notebook can read
   > `dataset/train/*.tsv` etc. from `/kaggle/input/mlchallenge2026-er/`.
3. Optionally set the upload visibility to "Only me" until you need to share.

### Google Colab

Two options:

- **Drive**: copy `student_resource/` into
  `MyDrive/amazon_ml/student_resource`, mount Drive in the notebook, then set
  `DATA_ROOT = "/content/drive/MyDrive/amazon_ml/student_resource"`.
- **Local upload**: run the notebook's upload helper cell, which places the
  files into `/content/amazon_ml/` (set `DATA_ROOT` to that folder).

## 2. Open the notebook

- **Kaggle**: Create a Notebook, then *File -> Upload Notebook* and pick
  `notebooks/amazon_ml_challenge_baseline.ipynb`. Use the default (or a 2x
  CPU / 16GB+ RAM) environment - GPU is NOT used by the baseline.
- **Colab**: *File -> Upload notebook* (or drag & drop), or open from Drive.

You must also make the `code/` modules visible. Easiest: upload
`code/business_entity_resolution/` into the runtime working dir (Kaggle:
`/kaggle/working/`; Colab: `/content/`). The notebook's **configure paths**
cell builds `CODE_ROOT` and adds `src/` to `sys.path`. If you put the code in
a different place, edit `CODE_ROOT` there.

## 3. Configure `DATA_ROOT`

The very first code cell auto-detects the environment:

- Kaggle: looks at each `/kaggle/input/<name>` and returns the first directory
  containing `dataset/` + `utils/`.
- Colab: returns `/content/amazon_ml` (or lets you paste a Drive path).
- Otherwise: falls back to the current working directory.

You can always override with:

```python
cfg = Config(data_root="/your/explicit/path")
```

`DATA_ROOT` is used for *everything* (`dataset/`, `utils/`, `cache/`,
`output/` derived as subfolders) - never hard-code Windows paths in the
notebook.

## 4. Dependencies

The **Setup** cell installs, inside the runtime only:

```
pip install -q pandas numpy scikit-learn scipy lightgbm rapidfuzz pyarrow tqdm
```

(already present on Kaggle/Colab; the install is a no-op mostly). The cell
then prints versions. `requirements.txt` in the code folder records the same
set for the submission package. Nothing is installed on the laptop.

## 5. Which cells to run first (order matters)

Run the notebook **top to bottom once**. The markdown section headers match
the numbered sections in the notebook. Cell dependencies:

1. **config + paths** (needs nothing)
2. **setup / install deps** (needs internet; ~1-2 min)
3. **import src modules**
4. **dataset loading** (reads all 7 TSVs, normalizes, caches to `cache/`)
5. **EDA summary** (prints counts; fast - uses the cache)
6. **validation split** (grouped by S1)
7. **blocking train** (builds TF-IDF index + candidates; slow first time)
8. **blocking recall** (reads the metrics)
9. **features train** (RapidFuzz pass; slow)
10. **train LightGBM**
11. **threshold tuning + validation report**
12. **one-parent effect**
13. **final model retrain** (optional)
14. **blocking test** -> **infer test** -> **write outputs** -> **validator**

The last cells print `VALIDATOR: PASS/FAIL`.

### Smoke mode

To verify the whole code path quickly (recommended on first run), set

```python
cfg.smoke = True
cfg.smoke_s1_limit = 4000
cfg.smoke_pool_limit = 60000
```

This uses a few thousand S1 rows per country and produces valid (tiny)
output files so you can confirm every stage works before the full run.

## 6. Expected runtime / size (full run, CPU)

Approximate, single-session guidance:

| Stage | Time | RAM |
|---|---|---|
| load + normalize + cache all entities | 10-25 min | 8-14 GB transient |
| blocking (build index + top-k per country) | 30-90 min | matrix ~3-6 GB / country |
| features (RapidFuzz) | 30-90 min | ~4-8 GB |
| LightGBM (+ tuning) | 5-20 min | ~4-8 GB |
| test blocking + inference + outputs | 30-90 min | ~6-10 GB |

Cache files: `cache/*.parquet` (entities ~ 2-4 GB; candidates ~ 0.5-2 GB).

> If RAM is tight: lower `max_features` (default 300k), raise `query_chunk`
> down (def. 64), lower `block_top_k` (def. 40) / `max_candidates_per_s1`
> (def. 25), or enable `cfg.light_features = True`.

Kaggle default sessions expire after 12 h of runtime; Colab free sessions
~ up to 12 h. The full baseline fits comfortably in one session if you keep
the SMOKE run first.

## 7. Where outputs are generated

Inside the runtime:

```
<DATA_ROOT>/output/
├── matching_results.tsv     # the ONLY leaderboard file
└── candidate_pairs.tsv      # blocking candidate set (matches ⊆ candidates)
```

The notebook prints the absolute paths after writing them.

## 8. Downloading the final submission

- **Kaggle**: both files are under `/kaggle/working/...`; Kaggle stores them
  in *Output* (Version tab) if you save a version, or use the file panel to
  download directly. `matching_results.tsv` is what you upload to the Portal.
- **Colab**: use the notebook's **save/download** cell
  (`files.download(...)`), or right-click in the file browser.

For the **final zip** (only after you also have validated outputs): zip
`output/`, `code/business_entity_resolution/` and the filled-in
`Documentation_template.md` per the PS.md structure. Do NOT include the
dataset or cache in the zip.

## 9. Resume after a runtime disconnect / session restart

Because every heavy artifact is cached under `cache/`:

1. Re-open the notebook (same notebook + dataset + code on the platform).
2. Re-run the **config**, **imports**, and **dataset loading** cells. Loading
   is instant after the first run: entities are read from `cache/*.parquet`.
3. Re-run **blocking**: candidates are also cached
   (`cache/candidates_train.parquet`, `cache/candidates_test.parquet`) and
   loaded instead of rebuilt, unless you set `cfg.cache_candidates = False`.
4. Pick up where you left off - the model, tuning and validation stages re-run
   quickly relative to the heavy first stages.

> Deleting `cache/` forces a full rebuild (low-disk warning: don't delete it
> while intermediate results are needed).

## 10. Compliance quick check (before any submission)

- No external lookups/Geocoding/registry - none of the code does this.
- `country` handled as an open set (France processed like any country).
- Model = LightGBM (MIT). No LLM/embeddings in this baseline.
- A3 empty-non-singleton convention is **provisional (0.0)** and printed as
  such; not claimed as an organizer rule.
- If uncertain about any rule, see `PS.md` $5/6/11 and flag, don't guess.