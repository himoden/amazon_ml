"""LightGBM training / inference wrapper.

CPU-friendly by design; ``lightgbm`` is imported lazily so the module can be
import-linted in any environment. Hard negatives are the default learning
signal (candidate pairs that look plausible but are not ground-truth links),
which is exactly what precision-hungry F0.5 needs.
"""

from __future__ import annotations

import numpy as np


def default_params(cfg, n_pos=None, n_neg=None, seed=42):
    params = dict(cfg.lgbm_params)
    params["seed"] = seed
    params["n_jobs"] = cfg.n_jobs
    if n_pos and n_neg and n_neg >= n_pos:
        params["scale_pos_weight"] = n_neg / n_pos
    return params


def train_lgbm(X, y, X_val, y_val, cfg, params=None, verbose=True):
    """Train LightGBM with early stopping on the validation split.

    Returns ``(model, best_iteration, history)``.
    The X arrays must be float32 numpy (numpy dtype coercion is applied here).
    """
    import lightgbm as lgb
    import time as _t

    t0 = _t.time()
    X = np.asarray(X, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    y = np.asarray(y)

    n_pos = max(int((y == 1).sum()), 1)
    n_neg = max(int((y == 0).sum()), 1)
    params = params or default_params(cfg, n_pos, n_neg, seed=cfg.seed)

    dtr = lgb.Dataset(X, label=y)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtr)

    callbacks = [
        lgb.early_stopping(cfg.early_stopping_rounds, verbose=False),
        lgb.log_evaluation(100),
    ]
    model = lgb.train(
        params,
        dtr,
        num_boost_round=cfg.num_boost_round,
        valid_sets=[dval],
        callbacks=callbacks,
    )
    if verbose:
        print(f"[model] LightGBM trained in {_t.time() - t0:.1f}s, "
              f"best_iteration={model.best_iteration}, "
              f"n_pos={n_pos}, n_neg={n_neg}")
    return model, getattr(model, "best_iteration", None), None


def predict_proba(model, X):
    """Calibrated-ish probability per candidate pair."""
    X = np.asarray(X, dtype=np.float32)
    it = getattr(model, "best_iteration", None)
    return model.predict(X, num_iteration=it)


def feature_importance(model, names, top=25):
    """Return a DataFrame of LightGBM gain importances (for the report)."""
    import pandas as pd

    imp = model.feature_importance(importance_type="gain")
    df = pd.DataFrame({"feature": names, "gain": imp}).sort_values("gain", ascending=False)
    if top:
        df = df.head(top)
    return df.reset_index(drop=True)