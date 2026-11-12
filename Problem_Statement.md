# Problem Statement: Market Regime Detection and Regime-Aware Risk Management for the Indian Equity Market

**Course:** DS605 Fundamentals of Machine Learning, M.Sc. Data Science (Autumn 2026)
**Team:** Pujita and Khushboo
**Domain:** Finance, Quantitative Risk Management
**Submission deadline:** Sun 15 Nov 2026

---

## 1. Background and Motivation

Stock markets do not behave the same way all the time. Some periods are calm, and some are stressed, such as 2008 and the 2020 Covid crash. Many simple investing rules ignore this and keep the same amount of money invested in every condition. This project asks a practical question: **can we recognize the type of market we are in directly from data, and does acting on that knowledge reduce risk?**

## 2. Problem Statement

Design, build and deploy a system that:

- **(a)** groups the trading days of the Indian equity market (March 2008 to October 2026) into **market regimes** (expected: Calm, Panic and Recovery) using **unsupervised learning** (PCA + K-Means, cross-checked with hierarchical clustering) on features built from returns, volatility, India VIX, stock correlation, currency, commodities and global markets,
- **(b)** predicts the regime of the **next trading day** using **supervised classification** (kNN, SVM, Random Forest and others), tested on a time-ordered train/test split and compared with a "tomorrow = today" baseline, and
- **(c)** tests whether a **simple regime-aware exposure rule** (hold less of the market in stressed regimes) gives better risk-adjusted performance (Sharpe ratio, maximum drawdown) than **buy-and-hold** on the Nifty 50, on out-of-sample data after transaction costs.

The problem is framed as **market-state detection and adaptive response**, not price prediction.

## 3. Key Terms

| Term | Meaning |
|---|---|
| **Market regime** | A type of market condition (for example calm, panic, recovery) with its own pattern of returns, volatility and correlation |
| **Volatility** | How widely daily returns swing (standard deviation of returns) |
| **India VIX** | An index of expected near-term volatility on the Nifty 50, often called the "fear gauge" |
| **Drawdown** | How far the market is below its previous peak |
| **Exposure** | The share of capital invested in the market (100% = fully invested, 0% = cash) |
| **Buy-and-hold** | Staying 100% invested the whole time (our baseline) |
| **Sharpe ratio** | Return above a safe rate divided by volatility: return earned per unit of risk |
| **Backtest** | Replaying a strategy on past data to estimate how it would have done |
| **Leakage / look-ahead** | Accidentally using future information when building or testing a model |
| **Persistence baseline** | A "model" that predicts tomorrow's regime to be the same as today's |

## 4. Objectives

1. Prepare one dataset from 11 core Indian and global series (Nifty 50, Sensex, Bank Nifty, Nifty IT, India VIX, USD/INR, Brent, gold, S&P 500, US 10Y yield, dollar index) and 15 large Indian stocks, enriched with the RBI repo rate.
2. Engineer about 14 features that describe the market each day.
3. Reduce them with PCA, discover regimes with K-Means (cross-checked with hierarchical clustering), choose the number of regimes with the silhouette score, and name each regime from its statistics.
4. Profile each regime (mean return, volatility, skewness, kurtosis, India VIX, drawdown) and check that known events fall in sensible regimes.
5. Train classifiers to predict the next-day regime and compare them with the persistence baseline.
6. Backtest buy-and-hold against the regime-aware exposure rule, with transaction costs, on the test period.
7. Deploy the results as an interactive web app.

## 5. Data

| Group | Source | Contents | Use |
|---|---|---|---|
| Core series (11) | Yahoo Finance (`yfinance`) | Indian indices, India VIX, USD/INR, Brent, gold, S&P 500, US 10Y, dollar index; 4,572 trading days, 3 Mar 2008 to 27 Oct 2026 | Regime features |
| Large stocks (15) | Yahoo Finance | Daily prices of 15 large Indian companies | Average correlation feature |
| Sector indices (6) | Yahoo Finance | Only start in 2010-2011 | Optional analysis, not in the core model |
| Macro enrichment | RBI (public repo-rate history) | RBI policy repo rate over time | Macro feature, safe-rate proxy for cash |

All data is public. Global series (S&P 500, US 10Y yield, dollar index, gold, Brent) are included because they influence Indian markets.

## 6. Scope and Originality

The techniques here (PCA, K-Means, hierarchical clustering, kNN, SVM, Random Forest) are standard and come from the course. This project does not claim a new ML method. Within a course project, it stands out through:

- an **Indian-market dataset assembled from several groups of series** (indices, India VIX, currency, commodities, global markets, stocks) plus an RBI macro enrichment, rather than a ready-made Kaggle file,
- an **end-to-end result**: regimes lead to a classifier, which leads to an exposure rule, and the backtested difference is reported rather than stopping at accuracy,
- **careful evaluation** (time-ordered splits, fitting on training data only, a persistence baseline, transaction costs),
- a **working deployment**.

## 7. Team Roles

| Member | Focus |
|---|---|
| Pujita | Features, PCA and clustering, regime profiling and validation, risk statistics per regime |
| Khushboo | Data pipeline and EDA, classifiers, backtest, deployment |
| Both | Understanding and explaining every step, README, viva |

## 8. Expected Outcomes

- A small set of interpretable regimes for Indian equity data that match known historical events
- Classifier results (accuracy, macro-F1) next to the persistence baseline
- A measured difference in Sharpe ratio and maximum drawdown between buy-and-hold and the regime-aware rule
- A deployed web app showing the current regime, regime history and backtest

## 9. How this meets the course requirements

| Requirement | How we meet it |
|---|---|
| Unique problem in a chosen domain | Regime detection for Indian equities with a measured risk outcome |
| Reliable public data | Yahoo Finance and RBI public data |
| No other group on the same dataset | To be confirmed with the TA; the combination of series and enrichment is our own |
| Enrich the dataset | RBI repo rate added to the market data |
| Every member can explain everything | Each phase has a lead and a reviewer; mock chalk talk before evaluation |

## 10. Limitations

- Regimes have no ground-truth labels, so classifier accuracy is measured against our own cluster labels.
- High-stress days are concentrated in two episodes (2008-09 and 2020), so regime counts and backtest differences have limited statistical weight.
- Historical backtests do not guarantee future results and depend on the rules, costs and time period chosen.
- The 15-stock list is today's large caps and is used only for the correlation feature.
- This is an educational course project and not investment advice.