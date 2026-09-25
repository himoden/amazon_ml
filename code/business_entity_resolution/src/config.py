"""Configuration for the Amazon ML Challenge 2026 pipeline.

Central place for every tunable knob. The notebook instantiates one
``Config`` and passes it through the pipeline stages. All paths are derived
from a single ``DATA_ROOT`` so the same code runs on Kaggle, Colab, or a
local copy without edits.

Compliance note
---------------
* ``country`` is an open set: nothing here hard-codes {US, India}.
* The dataset is processed WITHOUT external business/registry/geocoding data.
"""

from __future__ import annotations

import dataclasses
import os
import platform
import time


@dataclasses.dataclass
class Paths:
    """Resolved directories used by the pipeline."""

    data_root: str                      # where dataset/ and utils/ live
    dataset_dir: str
    cache_dir: str                      # normalized entities + candidate caches
    output_dir: str                     # final TSV outputs
    utils_dir: str

    @property
    def validator(self) -> str:
        """Path to utils/validate_submission.py."""
        return os.path.join(self.utils_dir, "validate_submission.py")


def detect_environment(data_root: str | None = None) -> tuple[str, str]:
    """Heuristically detect the runtime environment.

    Returns ``(resolved_data_root, environment_name)`` where
    ``environment_name`` in {"kaggle", "colab", "local"}.

    ``data_root`` is used verbatim when provided; otherwise we look for the
    standard locations for Kaggle/Colab and finally fall back to the current
    working directory.
    """
    # 1) explicit override ------------------------------------------------------------------
    if data_root:
        return data_root, "explicit"

    # 2) Kaggle: /kaggle/input contains the uploaded dataset(s) ------------------------------
    if os.path.isdir("/kaggle/input"):
        for name in sorted(os.listdir("/kaggle/input")):
            cand = os.path.join("/kaggle/input", name)
            if os.path.isdir(os.path.join(cand, "dataset")) and os.path.isdir(
                os.path.join(cand, "utils")
            ):
                return cand, "kaggle"
        # some users attach a differently laid-out dataset: pick the first dir with files
        for name in sorted(os.listdir("/kaggle/input")):
            cand = os.path.join("/kaggle/input", name)
            if os.path.isdir(cand):
                return cand, "kaggle"

    # 3) Colab: we expect the dataset to have been copied/uploaded to a folder --------------
    if os.path.isdir("/content"):
        for cand in ("/content/amazon_ml", "/content/amazon_ml_challenge",
                     "/content/dataset"):
            if os.path.isdir(cand):
                return cand, "colab"
        return "/content", "colab"

    # 4) Local fallback ----------------------------------------------------------------------
    return os.getcwd(), "local"


def default_paths(data_root: str, cache_dir: str | None = None,
                  output_dir: str | None = None) -> Paths:
    cache_dir = cache_dir or os.path.join(data_root, "cache")
    output_dir = output_dir or os.path.join(data_root, "output")
    return Paths(
        data_root=data_root,
        dataset_dir=os.path.join(data_root, "dataset"),
        cache_dir=cache_dir,
        output_dir=output_dir,
        utils_dir=os.path.join(data_root, "utils"),
    )


@dataclasses.dataclass
class Config:
    """All pipeline knobs. Defaults are tuned for a CPU remote runtime."""

    # --- seed --------------------------------------------------------------------------
    seed: int = 42

    # --- paths -------------------------------------------------------------------------
    data_root: str = ""
    cache_dir: str = "cache"
    output_dir: str = "output"

    # --- smoke test (tiny run to verify the code path end-to-end) -----------------------
    smoke: bool = False
    smoke_s1_limit: int = 4000          # S1 rows per country when smoke=True
    smoke_pool_limit: int = 60000       # S2/S3 rows per country when smoke=True

    # --- validation --------------------------------------------------------------------
    validation_fraction: float = 0.05   # fraction of S1 entities held out (grouped)
    full_validation: bool = False       # if True: validation_fraction is ignored
    required_fold: str = ""             # optional fixed fold name to resume

    # --- blocking ----------------------------------------------------------------------
    block_top_k: int = 40               # top-k TF-IDF retrieval candidates per S1
    max_candidates_per_s1: int = 25     # hard cap on the union candidate set per S1
    ngram_range: tuple = (3, 5)         # char n-gram range for TF-IDF
    max_features: int = 300_000         # TF-IDF vocabulary cap (memory control)
    min_df: int = 2
    query_chunk: int = 64               # retrieval chunk size (memory control)
    use_exact_keys: bool = True         # exact core-name key blocking pass
    use_numeric_pass: bool = False      # Future work: shared postal/numeric pass
    use_country_block: bool = True      # A2 confirmed in EDA: country is consistent
    percentile_floor: float = 0.0       # drop retrieval hits below this score (0 = off)

    # --- normalization -----------------------------------------------------------------
    transliterate_devanagari: bool = True   # manual, auditable rule (see normalize.py)
    keep_raw: bool = True               # always store raw name/address

    # --- features ----------------------------------------------------------------------
    light_features: bool = False        # cheaper feature set (trading accuracy for speed)
    char3_jaccard: bool = True          # address char-3-gram Jaccard feature
    contextual_features: bool = True    # rank / score-gap / n-candidates features
    n_jobs: int = os.cpu_count() or 1   # process pool for feature computation

    # --- modelling ---------------------------------------------------------------------
    model: str = "lightgbm"
    negative_sample_ratio: float = 1.0  # sampled negatives per positive (training)
    num_boost_round: int = 3000
    early_stopping_rounds: int = 100
    lgbm_params: dict = dataclasses.field(
        default_factory=lambda: dict(
            objective="binary",
            metric="binary_logloss",
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=100,
            feature_fraction=0.85,
            bagging_fraction=0.8,
            bagging_freq=1,
            lambda_l1=0.1,
            lambda_l2=1.0,
            n_jobs=-1,
            verbosity=-1,
            seed=42,
        )
    )

    # --- decision layer ------------------------------------------------------------------
    threshold_grid_first: tuple = (0.90, 0.92, 0.94, 0.95, 0.96, 0.97)
    threshold_grid_add: tuple = (0.94, 0.96, 0.98, 0.99)
    max_pred_per_s1: int = 12           # GT max seen is 11; stay a bit above it
    use_one_parent: bool = True         # A1 confirmed in EDA: apply global resolution

    # --- A3 handling --------------------------------------------------------------------
    # The official scorer behaviour for an empty prediction on a NON-singleton is not
    # documented (A3 UNVERIFIED). We use the provisional convention 0.0 and label it
    # in every report so it is never mistaken for an organizer rule.
    empty_non_singleton_score: float = 0.0

    # --- execution -----------------------------------------------------------------------
    cache_entities: bool = True         # parquet-cache normalized entity tables
    cache_candidates: bool = True       # parquet-cache candidate tables
    verbose: bool = True

    # ------------------------------------------------------------------ helpers -------
    def paths(self) -> Paths:
        """Resolve paths from data_root (requires self.data_root to be set)."""
        return default_paths(self.data_root, self.cache_dir, self.output_dir)

    def ensure_dirs(self) -> None:
        paths = self.paths()
        for d in (paths.cache_dir, paths.output_dir):
            os.makedirs(d, exist_ok=True)

    def env_summary(self) -> str:
        """One-line human-readable summary of the runtime."""
        lines = [
            "environment: " + platform.platform(),
            f"python: {platform.python_version()}",
            f"cpu: {os.cpu_count()} cores",
            "gpu: " + (os.environ.get("CUDA_VISIBLE_DEVICES", "n/a")),
        ]
        return "\n".join(lines)


def timer_report(start: float, tag: str = "") -> str:
    dt = time.time() - start
    m, s = divmod(int(dt), 60)
    h, m = divmod(m, 60)
    return f"[{tag}] elapsed: {h}h {m:02d}m {s:02d}s"