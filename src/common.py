# Paths, constants and small helpers shared by the notebooks and the app.
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
FIG = ROOT / "reports" / "figures"
for folder in (PROC, MODELS, FIG):
    folder.mkdir(parents=True, exist_ok=True)

SEED = 42
TD = 252  # trading days per year
TRAIN_END = pd.Timestamp("2017-12-31")  # fit on 2008-2017, test on 2018 onwards


def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=130, bbox_inches="tight")


def boundary_features(pcs, centres, raw_to_regime):
    """Distance from each day to every K-Means centre, the gap between the two nearest
    centres (small gap = a regime change is close) and the regime of the second-nearest one."""
    dist = np.linalg.norm(pcs.to_numpy()[:, None, :] - centres[None, :, :], axis=2)
    k = dist.shape[1]
    out = pd.DataFrame({f"dist_{raw_to_regime[j]}": dist[:, j] for j in range(k)}, index=pcs.index)
    out = out[[f"dist_{r}" for r in range(k)]]
    nearest = np.sort(dist, axis=1)
    out["margin"] = nearest[:, 1] - nearest[:, 0]
    second = pd.Series(np.argsort(dist, axis=1)[:, 1], index=pcs.index).map(raw_to_regime).astype(int)
    return out, second.rename("second_nearest")


def hold_filter(signal, n):
    """Only switch to a new value once it has held for n sessions in a row."""
    v = signal.to_numpy()
    out = v.copy()
    for t in range(1, len(v)):
        settled = t >= n - 1 and np.all(v[t - n + 1:t + 1] == v[t])
        out[t] = v[t] if settled else out[t - 1]
    return pd.Series(out, index=signal.index)


def regime_inputs(regimes, km, relabel, n_components):
    """Inputs for the change detectors: PCA scores, distances to the centres and today's regime (one-hot)."""
    pcs = regimes[[f"pc{i}" for i in range(1, n_components + 1)]]
    dist, second = boundary_features(pcs, km.cluster_centers_, relabel)
    today = pd.get_dummies(regimes["regime"]).reindex(columns=sorted(set(relabel.values())), fill_value=0)
    return pcs.join(dist).join(today.astype(float).add_prefix("cur_")), second


def run_backtest(signal, ret, rf, cost_bps, delay=1):
    """The signal at the close of day t sets the exposure of day t + delay.
    net = exposure * market + (1 - exposure) * cash - cost * |change in exposure|"""
    e = signal.shift(delay)
    e_prev = e.shift(1).fillna(e)  # the first position is free
    net = e * ret + (1 - e) * rf - cost_bps / 1e4 * (e - e_prev).abs()
    return net, e


def perf(net, e, rf):
    ok = net.dropna().index
    net, e, rf = net.loc[ok], e.loc[ok], rf.loc[ok]
    years = len(net) / TD
    wealth = (1 + net).cumprod()
    peak = np.maximum.accumulate(np.r_[1.0, wealth.to_numpy()])[1:]
    mdd = float((wealth.to_numpy() / peak - 1).min())
    cagr = float(wealth.iloc[-1] ** (1 / years) - 1)
    excess = net - rf
    trades = e.diff().abs().fillna(0)
    return {"CAGR %": 100 * cagr,
            "Volatility %": 100 * net.std() * np.sqrt(TD),
            "Sharpe": float(excess.mean() / excess.std() * np.sqrt(TD)),
            "Max drawdown %": 100 * mdd,
            "Calmar": cagr / abs(mdd),
            "Turnover (x/yr)": float(trades.sum() / years),
            "Position changes": int((trades > 1e-9).sum()),
            "Avg exposure": float(e.mean())}


def block_indices(n, rng, block=21):
    """Circular block bootstrap: resample runs of `block` consecutive days."""
    starts = rng.integers(0, n, int(np.ceil(n / block)))
    return ((starts[:, None] + np.arange(block)[None, :]) % n).ravel()[:n]


def paired_bootstrap(stat, a, b, B=2000, block=21, seed=SEED):
    """95% interval of stat(a) - stat(b), resampling the same days for both."""
    rng = np.random.default_rng(seed)
    diffs = np.empty(B)
    for i in range(B):
        idx = block_indices(len(a), rng, block)
        diffs[i] = stat(a, idx) - stat(b, idx)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), float((diffs > 0).mean())
