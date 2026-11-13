# Project Outline (starting from the raw data)
**Regime-Aware Adaptive Trading & Risk Management System**
DS605 | Team: Pujita and Khushboo | Plan date: Tue 10 Nov 2026 | **Deadline: Sun 15 Nov 2026** (late within 24 hours = 20% penalty)

Tiers: **T1** = must-have, **T2** = only if T1 is done early.
Suggested owners are in brackets: P = Pujita, K = Khushboo. Swap freely, but both must be able to explain everything in the viva.

---

## 1. The data we start with (raw)

| Group | Files | Used for | Known issues to handle |
|---|---|---|---|
| **Core (11)** | nifty50, sensex, banknifty, nifty_it, india_vix, usdinr, brent, gold, sp500, us10y, dxy | Regime features | Different holiday calendars; India VIX starts around Mar 2008; USD/INR glitch ticks; gold and Brent are futures (roll jumps) |
| **Sectors (6)** | nifty_auto, nifty_energy, nifty_fmcg, nifty_metal, nifty_pharma, nifty_realty | Optional features (post-2011 only) | Blank before mid-2011 (about 600-860 missing days each) |
| **Stocks (15)** | reliance, hdfcbank, icicibank, infy, tcs, itc, lt, sbin, bhartiartl, hindunilvr, kotakbank, axisbank, maruti, sunpharma, mahindra | Correlation feature, portfolio VaR, robustness | Chosen from today's large caps (survivorship bias) |

Plus **RBI repo rate and CPI inflation**, which we still have to scrape (our enrichment layer).

## 2. Scope

| Tier | Contents |
|---|---|
| **T1** | Data audit, merge, cleaning, EDA, returns, RBI enrichment, features, PCA, K-Means + hierarchical cross-check, regime validation, next-regime classifiers, regime-conditional VaR, backtest S0 and S1, Streamlit app, README |
| **T2** | Backtest S2 (VaR-based sizing), macro on/off ablation, basket portfolio VaR, classifier-driven backtest, sector features (post-2011), bootstrap confidence interval, stability check |
| **Future work** (mention in report only) | HMM/GMM, ML trading signal, XGBoost, Indian yield curve, FPI flows |

## 3. Key decisions (fixed now)

- **Calendar:** Nifty 50's trading days are the master calendar. Forward-fill gaps of **at most 3 days**.
- **Three separate tables** (core, sectors, stocks). Never merge sectors into core and call `dropna()`, or the 2008-2011 period disappears.
- **Start date:** where India VIX begins (about Mar 2008), so 2008 stays in.
- **Prices:** use `Close`, adjusted for splits and dividends (`auto_adjust=True`). Sensex is very highly correlated with Nifty, so use it only as a cross-check, not as a feature.
- **Returns:** log returns `r_t = ln(P_t / P_(t-1))`. For portfolio averages across stocks, convert to simple returns first (`exp(r) - 1`).
- **Windows:** returns 1/5/21 days, volatility 10/21/63 days, skewness and kurtosis 63 days. **No window longer than 63 days** (a 200-day window would remove the 2008 crash from the feature table).
- **Look-ahead rule:** anything computed from data up to the close of day *t* may only earn the return of day *t+1* or later.
- **Split:** fit scaler, PCA and clusters on 2008-2017 only; test on 2018 to Oct 2026 (confirm regime counts first). Test days are assigned with `KMeans.predict`.
- **Regimes:** test k = 2 to 5 (silhouette, minimum cluster size, interpretability). Name them from their statistics, never in advance.
- **Classifier target:** the regime at *t+1* from features at *t* (predicting today's regime would just re-learn the clustering). Always compare with the "tomorrow = today" baseline.
- **Risk-free rate:** choose one value (a flat assumed rate or the repo rate) and use it for both Sharpe and the cash leg.

## 4. Phases and tasks

### Phase 0: Setup (Tue 10 Nov)
- [ ] 0.1 Create the repo: `data/raw`, `data/interim`, `data/processed`, `notebooks`, `src`, `app`, `reports/figures` | T1 | [K]
- [ ] 0.2 Agree on Section 3 decisions in writing | T1 | [both]
- [ ] 0.3 Confirm that one of us registered finance/markets as a lab domain | T1 | [P]

### Phase 1: Data audit and merge (Tue 10 Nov)
- [ ] 1.1 Load all 32 raw files; print shape, date range, dtypes and missing counts per file | T1 | [K]
- [ ] 1.2 Record each series' start and end dates in a table (check India VIX and the sectors) | T1 | [K]
- [ ] 1.3 Build the core table: reindex the 11 series to Nifty's trading days, keep `Close` only | T1 | [K]
- [ ] 1.4 Build the sectors table and stocks table separately | T1 | [K]
- [ ] 1.5 Forward-fill gaps up to 3 days; count what is still missing | T1 | [K]
- [ ] 1.6 Save to `data/interim/` (3 files) | T1 | [K]

### Phase 2: Cleaning (Tue 10 Nov)
- [ ] 2.1 Trim the core table to the India VIX start date | T1 | [P]
- [ ] 2.2 Flag glitch ticks (very large one-day moves that reverse the next day, or Close outside High/Low) | T1 | [P]
- [ ] 2.3 Check every flagged day before fixing it. **Keep real extremes** (2008 crash, 23 Mar 2020) | T1 | [P]
- [ ] 2.4 Log every fix in `cleaning_log.csv` (date, series, old value, new value, reason) | T1 | [P]
- [ ] 2.5 Confirm zero missing values in core; save `data/processed/clean.csv` | T1 | [P]

### Phase 3: Returns and EDA (Wed 11 Nov)
- [ ] 3.1 Compute log returns; drop the first row; save `returns.csv` | T1 | [K]
- [ ] 3.2 Plots: prices (log scale), rolling volatility, correlation heatmap, return histograms with a normal curve, QQ plot (fat tails) | T1 | [K]
- [ ] 3.3 ADF test on prices vs returns. Wording: "returns have no unit root; variance is time-varying, which motivates regimes" | T1 | [K]
- [ ] 3.4 Summary table per series: mean, std, skewness, kurtosis | T1 | [K]

### Phase 4: Macro enrichment (Tue 10 to Wed 11 Nov)
- [ ] 4.1 Scrape or download the RBI repo rate history | T1 | [P]
- [ ] 4.2 Scrape or download CPI inflation (YoY) | T1 | [P]
- [ ] 4.3 Convert to daily dates using the **release date**, then forward-fill between releases | T1 | [P]
- [ ] 4.4 Save `data/processed/macro.csv` | T1 | [P]

### Phase 5: Feature engineering (Wed 11 Nov)
- [ ] 5.1 Nifty returns (1/5/21), volatility (10/21/63, annualised by sqrt(252)), drawdown from running peak, 21/63-day moving-average gap | T1 | [K]
- [ ] 5.2 Rolling skewness and kurtosis (63 days) | T1 | [P]
- [ ] 5.3 India VIX level and 5-day change; Bank Nifty minus Nifty 21-day return | T1 | [K]
- [ ] 5.4 USD/INR, Brent, gold, S&P 500 21-day returns; US 10Y level | T1 | [K]
- [ ] 5.5 Average pairwise correlation of the 15 stocks (21 or 63 days) | T1 | [P]
- [ ] 5.6 Merge macro features; check there is no look-ahead; save `features.csv` | T1 | [both]

### Phase 6: PCA and regime discovery (Wed 11 to Thu 12 Nov)
- [ ] 6.1 Split train/test; fit `StandardScaler` and PCA on train only (keep about 85-90% variance) | T1 | [P]
- [ ] 6.2 K-Means for k = 2 to 5; record silhouette and cluster sizes | T1 | [P]
- [ ] 6.3 Hierarchical (Ward) cross-check and dendrogram | T1 | [P]
- [ ] 6.4 Choose k and write down why; name the regimes from their statistics | T1 | [both]
- [ ] 6.5 Assign test days with `KMeans.predict`; save `regimes.csv` | T1 | [P]

### Phase 7: Regime validation (Thu 12 Nov)
- [ ] 7.1 Known-events table (2008, 2011, 2013, 2020, 2022) vs assigned regime | T1 | [K]
- [ ] 7.2 Per-regime table: mean return, volatility, VIX, number of days | T1 | [K]
- [ ] 7.3 Transition matrix and average regime duration | T1 | [K]
- [ ] 7.4 Stability check with a different training window | T2 | [K]

### Phase 8: Next-regime classifier (Thu 12 to Fri 13 Nov)
- [ ] 8.1 Build the target (regime at t+1 from features at t) | T1 | [K]
- [ ] 8.2 Persistence baseline; Logistic Regression, kNN, SVM, Random Forest with `TimeSeriesSplit` | T1 | [K]
- [ ] 8.3 `VotingClassifier` of the best models | T1 | [K]
- [ ] 8.4 Results table: accuracy, macro-F1, confusion matrix vs the baseline | T1 | [K]
- [ ] 8.5 Save the model, scaler and PCA with `joblib` | T1 | [K]
- [ ] 8.6 Ablation with vs without macro features | T2 | [P]

### Phase 9: Risk overlay and backtest (Thu 12 to Fri 13 Nov)
- [ ] 9.1 1-day 95% historical-simulation VaR of Nifty per regime, train data only | T1 | [P]
- [ ] 9.2 VaR breach rate on test data (should be near 5%) | T1 | [P]
- [ ] 9.3 **S0:** buy-and-hold Nifty | T1 | [K]
- [ ] 9.4 **S1:** exposure by current regime (calm 100%, middle 50%, panic cash); regime at close of *t* sets exposure for *t+1* | T1 | [K]
- [ ] 9.5 Transaction costs with 0 / 10 / 25 bps sensitivity | T1 | [K]
- [ ] 9.6 Metrics: CAGR, volatility, Sharpe, max drawdown, turnover; results inside 2020 and 2022 reported separately | T1 | [P]
- [ ] 9.7 **S2:** VaR-based exposure = VaR(calm) / VaR(regime), capped at 100% | T2 | [P]
- [ ] 9.8 Basket portfolio VaR per regime | T2 | [P]

### Phase 10: Deployment (Sat 14 Nov)
- [ ] 10.1 Streamlit pages: current regime, regime timeline, VaR table, backtest chart | T1 | [K]
- [ ] 10.2 Load the saved model with `joblib` (no retraining on load) | T1 | [K]
- [ ] 10.3 `requirements.txt`; deploy on Streamlit Community Cloud; test the URL in a clean browser | T1 | [K]
- [ ] 10.4 Optional live `yfinance` refresh with fallback to cached CSVs | T2 | [K]

### Phase 11: Documentation and viva (Sat 14 Nov onward)
- [ ] 11.1 README: dataset description with scale (about 32 series x 4,500 days), pipeline, how to run | T1 | [P]
- [ ] 11.2 Limitations: hindsight labels, few independent crises, survivorship in the basket, no guarantee of future performance | T1 | [P]
- [ ] 11.3 Update `Problem_Statement.md` (move the ML trading signal to future work; list final data sources) | T1 | [P]
- [ ] 11.4 Fresh-clone test; submit before the deadline | T1 | [both]
- [ ] 11.5 Viva: each of us explains every file, function and library; mock chalk talk where you present each other's part | T1 | [both]

## 5. Timeline

| Day | Focus | End-of-day check |
|---|---|---|
| **Tue 10** | Phases 0-2 (setup, audit, merge, cleaning); start Phase 4 | Clean core table with a cleaning log |
| **Wed 11** | Phases 3-5 (returns, EDA, features); start Phase 6 | `features.csv` with no look-ahead |
| **Thu 12** | Phases 6-7 (regimes, validation); start Phases 8-9 | **Checkpoint:** validated regimes and VaR table |
| **Fri 13** | Phase 8 results, Phase 9 backtest | Comparison table and plots |
| **Sat 14** | Phases 10-11 (deploy, README) | Public URL works |
| **Sun 15** | Fresh-clone test, submit | Submitted |
| **Mon 16+** | Viva rehearsal | Both can explain everything |

## 6. Cut order (if time runs out)

1. All T2 tasks.
2. The `VotingClassifier` and the hierarchical cross-check.
3. The RBI enrichment (declare it as future work).
4. Fewer features and a fixed k.

**Never cut:** the leakage rules (train-only fitting, the t to t+1 rule, time-ordered splits), the persistence baseline, event validation, and the deployed app.

## 7. Tough questions to rehearse

- Your labels come from your own clustering. What does classifier accuracy mean?
- How do you know the regimes are real, not arbitrary partitions?
- Did you leak the future anywhere?
- Is the improvement significant with so few crises?
- What happens to the result with transaction costs?
- Why log returns, and why not the raw prices?