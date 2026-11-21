"""Streamlit app: regime-aware risk dashboard for the Nifty 50 (DS605 project).

Run from the repo root:   streamlit run app.py
The app only LOADS the frozen Phase 6 model (joblib) and the saved result tables. It never retrains and
never downloads data, so every number on screen can be traced to a file in data/processed/.
"""
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import ListedColormap

from src.inference import (CALM, DECLINE, REBOUND, PROC, S1_EXPOSURE, hold_filter, integrity_check,
                           load_artifacts, predict_regimes)

st.set_page_config(page_title="Regime-Aware Risk Dashboard", page_icon="📉", layout="wide")

COLORS = {CALM: "#2a9d8f", DECLINE: "#e63946", REBOUND: "#f4a261"}
STRATS = {"S0 Buy-and-hold": "S0", "S1 Regime 100/50/0": "S1_N1", "S1 with 2-session persistence filter": "S1_N2", "A VIX rule (rival)": "A_N1"}
TD = 252


# ---------------------------------------------------------------- data (cached) ----
@st.cache_resource
def get_artifacts():
    return load_artifacts()


@st.cache_data
def read(name, **kw):
    return pd.read_csv(PROC / name, **kw)


@st.cache_data
def get_regimes():
    art = get_artifacts()
    feats = pd.read_csv(PROC / "features.csv", index_col="Date", parse_dates=True)
    saved = pd.read_csv(PROC / "regimes.csv", index_col="Date", parse_dates=True)
    live = predict_regimes(feats, art)                                 # saved scaler -> PCA -> K-Means, no refit
    ok, mism, n = integrity_check(live, saved["regime"])
    exposure_signal = live.map(S1_EXPOSURE)
    return {"feats": feats, "saved": saved, "live": live, "ok": ok, "mism": mism, "n": n,
            "s1": exposure_signal, "s1_filtered": hold_filter(exposure_signal, 2)}


art, R = get_artifacts(), get_regimes()
NAMES = art["names"]
live, saved, feats = R["live"], R["saved"], R["feats"]
last_date = live.index[-1]
var_tab = read("backtest_var_by_regime.csv", index_col="regime")

# ---------------------------------------------------------------- sidebar ----
with st.sidebar:
    st.header("About")
    st.write("Regime-aware trading and risk management for Indian equities (DS605 course project).")
    st.markdown(f"**Data through:** {last_date.date()}  \n**Model:** K-Means, k = 3, {art['meta']['n_components']} PCA components "
                f"({art['meta']['explained_variance']:.1%} variance), fitted once on data up to {art['meta']['train_end']} and frozen.")
    if R["ok"]:
        st.success(f"Integrity check passed: the saved model reproduces the Phase 6 labels on the last {R['n']} sessions.")
    else:
        st.error(f"Integrity check FAILED: {R['mism']} of the last {R['n']} labels differ from regimes.csv. "
                 "The installed scikit-learn version may differ from the one that saved the model.")
    st.caption("Educational project. Not investment advice. Past performance does not guarantee future results.")

st.title("Regime-Aware Adaptive Trading and Risk Management")
st.caption("Nifty 50 | market regimes found by PCA and K-Means | risk overlay tested out of sample, 2018 to 2026")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Current regime", "Regime timeline", "Risk (VaR)", "Backtest", "Evidence and limits"])

# ================================================================= tab 1 ====
with tab1:
    reg_now = int(live.iloc[-1])
    age = (date.today() - last_date.date()).days
    if age > 4:
        st.warning(f"The data ends on {last_date.date()}, {age} days ago. This page shows the regime at that close, "
                   "not today's. The app does not download new data.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Regime at the last close", NAMES[reg_now])
    c2.metric("India VIX", f"{saved['vix_level'].iloc[-1]:.1f}")
    c3.metric("Nifty drawdown from peak", f"{saved['nifty_drawdown'].iloc[-1] * 100:.1f}%")
    c4.metric("Nifty close", f"{saved['nifty_close'].iloc[-1]:,.0f}")

    st.subheader("What the rule says for the next session")
    e1, e2 = float(R["s1"].iloc[-1]), float(R["s1_filtered"].iloc[-1])
    v_next = var_tab.loc[NAMES[reg_now], "VaR 95% (next day) %"]
    e_next = var_tab.loc[NAMES[reg_now], "ES 95% (next day) %"]
    a, b, c = st.columns(3)
    a.metric("Exposure, plain S1", f"{e1:.0%}", help="Calm 100%, Stressed-Rebound 50%, Stressed-Decline cash. The regime at the close of day t sets the exposure for day t+1.")
    b.metric("Exposure, with persistence filter", f"{e2:.0%}", help="Changes position only after the new regime has held for 2 sessions.")
    c.metric("Next-day 95% VaR after this regime", f"{v_next:.2f}%", help=f"Historical simulation on the training period. Expected shortfall: {e_next:.2f}%.")
    if e1 != e2:
        st.info("The filter is still holding the earlier position because the new regime has not yet lasted 2 sessions.")
    st.caption("Whether regime-based exposure improves returns has **not** been demonstrated (see the Evidence tab). "
               "What the regime does show reliably is how large the next move is likely to be.")

    st.subheader("Last 20 sessions")
    last = pd.DataFrame({"Regime": live.map(NAMES), "India VIX": saved["vix_level"], "Nifty close": saved["nifty_close"].round(0),
                         "Exposure (S1)": R["s1"].map("{:.0%}".format), "Exposure (filtered)": R["s1_filtered"].map("{:.0%}".format)}).tail(20).iloc[::-1]
    last.index = last.index.date
    st.dataframe(last, width="stretch")

# ================================================================= tab 2 ====
with tab2:
    st.write("Regime of every session. Labels are known at the close of each day; they lag the market (the first stressed label tends to come after the Nifty has already fallen).")
    yrs = sorted(set(saved.index.year))
    y0, y1 = st.slider("Years shown", yrs[0], yrs[-1], (yrs[0], yrs[-1]))
    wf = read("robust_daily_returns.csv", index_col="Date", parse_dates=True)["regime_walkforward"]
    show_wf = st.checkbox("Also show the walk-forward labels (model refitted each January; test years only)", value=False)
    m = (saved.index.year >= y0) & (saved.index.year <= y1)
    d, lab = saved[m], live[m]
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.plot(d.index, d["nifty_close"], color="#555555", lw=0.8)
    for k in (CALM, DECLINE, REBOUND):
        sel = lab == k
        ax.scatter(d.index[sel], d["nifty_close"][sel], s=5 if k == CALM else 14, color=COLORS[k], label=NAMES[k], zorder=3)
    ax.axvline(pd.Timestamp(art["meta"]["train_end"]), color="black", ls="--", lw=1)
    ax.text(pd.Timestamp(art["meta"]["train_end"]), ax.get_ylim()[1] * 0.98, "  end of training data", va="top", fontsize=8)
    ax.set_yscale("log"); ax.set_ylabel("Nifty 50 (log scale)"); ax.legend(loc="upper left", markerscale=2)
    st.pyplot(fig); plt.close(fig)
    if show_wf:
        wfm = wf[(wf.index.year >= max(y0, 2018)) & (wf.index.year <= y1) & (wf.index > pd.Timestamp(art["meta"]["train_end"]))]
        if len(wfm):
            fz = live.reindex(wfm.index)
            fig, ax = plt.subplots(figsize=(12, 1.4))
            ax.imshow(np.vstack([fz.to_numpy(), wfm.to_numpy()]), aspect="auto", cmap=ListedColormap([COLORS[0], COLORS[1], COLORS[2]]), vmin=0, vmax=2, interpolation="nearest")
            ax.set_yticks([0, 1]); ax.set_yticklabels(["frozen", "walk-forward"]); ax.set_xticks([])
            st.pyplot(fig); plt.close(fig)
            st.caption(f"The two labelings agree on {(fz == wfm).mean():.1%} of the days shown, but most days are Calm. They differ mostly in 2025 to 2026.")
        else:
            st.caption("Walk-forward labels exist for the test years (2018 onward) only.")
    share = pd.DataFrame({"Share of sessions %": (lab.value_counts(normalize=True).reindex([0, 1, 2]).fillna(0) * 100).round(1)}); share.index = [NAMES[i] for i in share.index]
    ep = read("regime_duration_summary.csv", index_col=0).round(1)
    cc1, cc2 = st.columns(2)
    cc1.write("**Share of sessions in the selected years**"); cc1.dataframe(share, width="stretch")
    cc2.write("**How long each regime lasts (Phase 7)**"); cc2.dataframe(ep, width="stretch")

# ================================================================= tab 3 ====
with tab3:
    st.write("One-day 95% **historical-simulation** VaR of the Nifty, given the regime at the previous close. "
             "Computed on the training period (2008 to 2017) only, then checked on the 2018 to 2026 test period.")
    show = var_tab.rename(columns={"days (train, next-day)": "Days", "mean next-day return %": "Mean next-day return %", "next-day volatility % (daily)": "Daily volatility %",
                                   "VaR 95% (next day) %": "VaR 95%", "ES 95% (next day) %": "Expected shortfall 95%"}).drop(columns=["VaR 95% (same day, reference) %"])
    st.dataframe(show.round(2), width="stretch")
    br = read("backtest_var_breaches_test.csv")
    br.columns = [c if not c.startswith("next-day regime") else "Group" for c in br.columns]
    st.write("**Test-period breach rate** (target about 5% of days; the interval shows what is statistically compatible with 5%)")
    st.dataframe(br.round(2), width="stretch", hide_index=True)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    names = list(var_tab.index)
    bar_col = {NAMES[k]: COLORS[k] for k in COLORS}
    ax.bar(names, var_tab["VaR 95% (next day) %"], color=[bar_col.get(n, "#888888") for n in names], edgecolor="black")
    ax.set_ylabel("Next-day VaR 95%, % of the index"); plt.xticks(rotation=15)
    st.pyplot(fig); plt.close(fig)
    st.caption("A stressed label means the next day's loss threshold is about two to three times larger, which is the robust finding. "
               "The VaR is conservative after Calm days (breach rate below 5%) but was breached on more than 5% of the few Stressed-Decline days (63 days, so the interval is wide).")

# ================================================================= tab 4 ====
with tab4:
    daily = read("robust_daily_returns.csv", index_col="Date", parse_dates=True)
    metrics = read("backtest_metrics.csv")
    c1, c2, c3 = st.columns(3)
    period = c1.selectbox("Period", ["Test 2018-2026 (out of sample)", "Training 2008-2017 (in sample)"])
    cost = c2.selectbox("Transaction cost (bps per unit traded)", [10, 0, 25])
    chosen = c3.multiselect("Strategies in the chart", list(STRATS), default=list(STRATS)[:3])
    split = "test" if period.startswith("Test") else "train"
    dd = daily[daily["split"] == split]
    st.write("**Metrics from the saved backtest (Phase 9)**")
    mt = metrics[(metrics["period"] == period) & (metrics["cost bps"] == cost)].set_index("strategy")
    st.dataframe(mt[["CAGR %", "Volatility % (ann.)", "Sharpe", "Max drawdown %", "Calmar", "Turnover (x per year)", "Average exposure"]].round(2), width="stretch")
    st.caption("S1 and the filtered version are shown in the chart at 10 bps (the headline cost); the table above responds to the cost selector. "
               "The filtered version is from Phase 9b and is not in the Phase 9 table.")
    if chosen:
        fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
        for s in chosen:
            r = dd[f"net_return_{STRATS[s]}_10bps"]
            w = (1 + r).cumprod(); ax[0].plot(w.index, w, label=s, lw=1.4)
            ax[1].plot(w.index, w / w.cummax() - 1, lw=1.2)
        ax[0].set_yscale("log"); ax[0].set_title("Growth of 1 (net of 10 bps)"); ax[0].legend(fontsize=8); ax[1].set_title("Drawdown from the previous peak")
        st.pyplot(fig); plt.close(fig)
    fm = read("robust_filter_metrics.csv")
    st.write("**Persistence filter, test period, 10 bps (Phase 9b)**")
    ft = fm[fm["period"] == "test"].set_index("strategy")[["CAGR %", "Sharpe", "Max drawdown %", "Position changes", "Turnover (x per year)"]]
    st.dataframe(ft.round(2), width="stretch")

# ================================================================= tab 5 ====
with tab5:
    st.subheader("What the evidence supports")
    st.markdown("""
- **Regimes describe risk well.** After a stressed label the next day's volatility is 2 to 4 times the Calm level, under the frozen model and under yearly refits.
- **The overlay reduces the damage of a crash like 2020.** Maximum drawdown is roughly 40 to 50% smaller than buy-and-hold, and this survives yearly refitting.
- **A better Sharpe ratio is not demonstrated.** Outside 2020 the regime strategy is worse than buy-and-hold, and no Sharpe difference is statistically distinguishable from zero.
- **The labels are not stable forever.** Refitting each year changes what the stressed regimes mean, especially after 2020 and in 2025 to 2026.
- **No classifier beats persistence** (tomorrow's regime equals today's) at predicting the next regime.
""")
    ex = read("robust_ex2020_metrics.csv")
    st.write("**Is the benefit just 2020?** Test period with events removed (10 bps)")
    st.dataframe(ex[ex["strategy"].isin(["S0", "S1", "A"])][["scenario", "strategy", "Sharpe", "Max drawdown %", "Position changes"]].round(2), width="stretch", hide_index=True)
    wfm = read("robust_walkforward_metrics.csv", index_col=0)
    st.write("**Walk-forward regimes versus frozen regimes** (test period, 10 bps)")
    st.dataframe(wfm[["CAGR %", "Sharpe", "Max drawdown %", "Position changes"]].round(2), width="stretch")
    st.write("**Experiment log** (rules were fixed before each result was computed)")
    st.dataframe(read("robust_experiment_log.csv"), width="stretch", hide_index=True)
    st.subheader("Limitations")
    st.markdown("""
- Regime labels are created by clustering with hindsight and are partly defined by volatility, so their link to risk is expected.
- Only a few independent crises exist in the sample (2008, 2020, a handful of smaller episodes).
- The 2018 to 2026 test period was used in Phase 9 and again for the robustness checks, so it is no longer a fresh sample.
- Index returns exclude dividends; cash earns the RBI repo rate (a proxy); transaction costs are a flat assumption.
- The stock basket used for risk work consists of today's large caps (survivorship bias).
- The app shows cached data and does not update itself. It makes no forecast of direction and gives no investment advice.
""")
