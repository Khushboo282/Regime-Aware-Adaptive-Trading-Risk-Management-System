"""Load the frozen Phase 6 regime model and apply it to features. Nothing is refitted here.

The Streamlit app (app.py) uses these functions; they can also be run from a notebook:
    from src.inference import load_artifacts, predict_regimes
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC, MODELS = ROOT / "data" / "processed", ROOT / "models"

CALM, DECLINE, REBOUND = 0, 1, 2
S1_EXPOSURE = {CALM: 1.0, REBOUND: 0.5, DECLINE: 0.0}      # Phase 9 rule: regime at the close of t sets exposure for t+1


def load_artifacts():
    """Scaler, PCA and K-Means saved in Phase 6 (loaded, never refitted) plus the label metadata."""
    scaler, pca, km = (joblib.load(MODELS / f"{n}.joblib") for n in ("scaler", "pca", "kmeans"))
    with open(PROC / "regime_meta.json") as f:
        meta = json.load(f)
    relabel = {int(k): int(v) for k, v in meta["relabel_from_kmeans_raw"].items()}
    names = {int(k): v for k, v in meta["regime_names"].items()}
    return {"scaler": scaler, "pca": pca, "kmeans": km, "relabel": relabel, "names": names, "meta": meta}


def predict_regimes(features, art):
    """Regime (0 Calm, 1 Stressed-Decline, 2 Stressed-Rebound) for every row of `features`, using only saved objects."""
    X = features[art["meta"]["feature_cols"]]
    Z = pd.DataFrame(art["scaler"].transform(X), index=X.index, columns=X.columns)    # keep feature names, as in Phase 6
    raw = art["kmeans"].predict(art["pca"].transform(Z))
    return pd.Series(raw, index=features.index, name="regime").map(art["relabel"]).astype(int)


def hold_filter(s, n):
    """Persistence filter from Phase 9b: a change is acted on only after the new value has held for n sessions up to day t."""
    v = s.to_numpy(); out = v.copy()
    for t in range(1, len(v)):
        if t >= n - 1 and np.all(v[t - n + 1:t + 1] == v[t]) and v[t] != out[t - 1]:
            out[t] = v[t]
        else:
            out[t] = out[t - 1]
    return pd.Series(out, index=s.index)


def integrity_check(live, saved, last_n=750):
    """Compare regimes recomputed from the saved model with the Phase 6 file on the last `last_n` sessions."""
    idx = saved.index[-last_n:]
    mism = int((live.reindex(idx) != saved.loc[idx]).sum())
    return mism == 0, mism, len(idx)
