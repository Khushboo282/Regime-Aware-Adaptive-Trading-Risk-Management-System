# Problem Statement: Regime-Aware Adaptive Trading & Risk Management System

**Course:** DS605 Fundamentals of Machine Learning, M.Sc. Data Science (Autumn 2026) <br>
**Domain:** Finance, Quantitative Risk Management & Algorithmic Trading <br>
**Submission deadline:** 15 Nov 2026

---

## 1. Background and Motivation

Trading and risk systems often apply a fixed strategy and a fixed risk tolerance regardless of prevailing market conditions. This is a widely recognized concern in quantitative risk management: strategies and risk limits calibrated during calm periods can perform poorly when volatility regimes shift sharply, as seen during past stress episodes such as 2008 and March 2020.

This project explores whether such regime shifts can be **detected directly from market data**, and whether that detection can be used to **adjust trading behavior and risk exposure** accordingly.

## 2. Problem Statement

Design and deploy a system that:

- **(a)** identifies latent market regimes (for example Calm/Trending, Choppy/Uncertain, High-Volatility/Panic, Recovery) from historical Indian equity market data using **unsupervised learning**,
- **(b)** predicts the prevailing regime using **supervised classification**, and
- **(c)** evaluates whether **regime-aware position sizing and risk limits** produce better risk-adjusted performance than a **regime-blind baseline strategy**, on backtested historical data.

The problem is framed as a **market-state detection and adaptive-response problem**, not direct price prediction, since forecasting returns directly is both harder and less reliable with the classical ML techniques available in this course.

## 3. Key Terms

| Term | Meaning |
|---|---|
| **Market regime** | A distinct market "state" (for example calm, choppy, panic, recovery) characterized by patterns in returns, volatility and correlation |
| **Latent** | Not directly labeled in the data; must be discovered by the model |
| **Position sizing** | How much capital is allocated to a trade |
| **Risk limits** | Rules that cap potential losses (for example Value-at-Risk thresholds) |
| **Regime-aware strategy** | Adjusts position size and risk limits based on the predicted regime |
| **Regime-blind baseline** | Uses the same position size and risk limits in all market conditions |
| **Risk-adjusted performance** | Return judged relative to the risk taken, measured by Sharpe ratio and maximum drawdown |
| **Backtest** | Replaying a strategy on historical data to estimate how it would have performed |

## 4. Objectives

1. Build a multi-asset, multi-source financial feature panel (returns, statistical moments, volatility, correlation, macro indicators).
2. Discover market regimes using unsupervised learning (K-Means, Hierarchical Clustering), validated via silhouette analysis.
3. Train supervised classifiers (kNN baseline, SVM, ensemble methods) to predict the prevailing regime from current features.
4. Build a rule-based risk overlay that adjusts position size and Value-at-Risk thresholds based on the predicted regime.
5. Build a classical ML trading signal using engineered technical features.
6. Backtest regime-aware vs regime-blind strategy performance and report the difference.
7. Deploy the resulting pipeline as an interactive web application.

## 5. Scope and Originality

The individual techniques (clustering, classification, VaR estimation) are standard and well established. This project does **not** claim a new ML method. Within the scope of a course project, it differentiates itself through:

- **A self-assembled, multi-source dataset:** equity indices, a volatility index, cross-sector correlation, commodities, and manually scraped macroeconomic indicators from RBI's public database.
- **An end-to-end pipeline that reaches a measured outcome:** regime detection feeds supervised forecasting, which feeds a concrete decision (position sizing and risk limits), and the backtested performance difference is reported explicitly.
- **Two complementary, individually explainable sub-systems:** a risk-overlay component and a trading-signal component.
- **A working deployment,** rather than a notebook-only submission.

## 6. Success Criteria / Expected Outcomes

- A set of market regimes for Indian equity data that are interpretable against known historical events
- A regime classifier with reported accuracy and F1 metrics
- A measured difference in risk-adjusted return (Sharpe ratio) between regime-aware and fixed-baseline position sizing
- A deployed, interactive demonstration of the complete pipeline

## 7. Limitations

- Regimes are discovered without ground-truth labels, so classifier "accuracy" is measured against the project's own cluster labels.
- Historical backtests do not guarantee future performance, and results can be sensitive to rules, costs, and time periods.
- This is an educational course project and does not constitute investment advice.
