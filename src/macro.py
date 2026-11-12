"""Macroeconomic Enrichment (RBI Repo Rate).

Creates data/processed/macro.csv aligned to the Nifty 50 calendar from clean.csv.

Usage:
    python -m src.macro
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = ROOT / "data" / "processed" / "clean.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "macro.csv"

# Official RBI Monetary Policy Repo Rate decisions (effective dates)
# Source: Reserve Bank of India (RBI DBIE / Policy Announcements 2008-2026)
RBI_REPO_HISTORY = [
    ("2008-01-01", 7.75),
    ("2008-06-12", 8.00),
    ("2008-06-25", 8.50),
    ("2008-07-30", 9.00),
    ("2008-10-20", 8.00),  # GFC emergency cuts begin
    ("2008-11-03", 7.50),
    ("2008-12-08", 6.50),
    ("2009-01-05", 5.50),
    ("2009-03-04", 5.00),
    ("2009-04-21", 4.75),
    ("2010-03-19", 5.00),  # Post-crisis hiking cycle
    ("2010-04-20", 5.25),
    ("2010-07-02", 5.50),
    ("2010-07-27", 5.75),
    ("2010-09-16", 6.00),
    ("2010-11-02", 6.25),
    ("2011-01-25", 6.50),
    ("2011-03-17", 6.75),
    ("2011-05-03", 7.25),
    ("2011-06-16", 7.50),
    ("2011-07-26", 8.00),
    ("2011-09-16", 8.25),
    ("2011-10-25", 8.50),
    ("2012-04-17", 8.00),
    ("2013-01-29", 7.75),
    ("2013-03-19", 7.50),
    ("2013-05-03", 7.25),
    ("2013-09-20", 7.50),  # Taper tantrum / Rajan hikes
    ("2013-10-29", 7.75),
    ("2014-01-28", 8.00),
    ("2015-01-15", 7.75),  # Easing cycle
    ("2015-03-04", 7.50),
    ("2015-06-02", 7.25),
    ("2015-09-29", 6.75),
    ("2016-04-05", 6.50),
    ("2016-10-04", 6.25),
    ("2017-08-02", 6.00),
    ("2018-06-06", 6.25),
    ("2018-08-01", 6.50),
    ("2019-02-07", 6.25),  # Pre-Covid easing
    ("2019-04-04", 6.00),
    ("2019-06-06", 5.75),
    ("2019-08-07", 5.40),
    ("2019-10-04", 5.15),
    ("2020-03-27", 4.40),  # Covid crash emergency cut
    ("2020-05-22", 4.00),
    ("2022-05-04", 4.40),  # Global inflation response hikes
    ("2022-06-08", 4.90),
    ("2022-08-05", 5.40),
    ("2022-09-30", 5.90),
    ("2022-12-07", 6.25),
    ("2023-02-08", 6.50),  # last hike of the 2022-23 cycle; held until Feb 2025
    ("2025-02-07", 6.25),
    ("2025-04-09", 6.00),
    ("2025-06-06", 5.50),
    ("2025-12-05", 5.25),
    ("2026-10-07", 5.50),  # first hike since Feb 2023
]



def build_macro_dataset():
    if not CLEAN_PATH.exists():
        raise FileNotFoundError(f"Missing {CLEAN_PATH}. Run Phase 2 first!")

    # 1. Read clean.csv to get the exact trading calendar
    clean_df = pd.read_csv(CLEAN_PATH, index_col=0)
    master_dates = pd.to_datetime(clean_df.index)

    # 2. Convert RBI historical list into a DataFrame
    rbi_df = pd.DataFrame(RBI_REPO_HISTORY, columns=["Date", "repo_rate"])
    rbi_df["Date"] = pd.to_datetime(rbi_df["Date"])
    rbi_df = rbi_df.set_index("Date").sort_index()

    # 3. Create full daily calendar starting from the earliest RBI date (2008-01-01)
    earliest_date = min(rbi_df.index[0], master_dates[0])
    full_date_range = pd.date_range(
        start=earliest_date, end=master_dates[-1], freq="D"
    )

    # Reindex and forward-fill: rate stays active until next change
    daily_rbi = rbi_df.reindex(full_date_range).ffill()

    # 4. Now filter strictly for Nifty 50's 4,572 trading days
    macro_df = daily_rbi.reindex(master_dates)

    # 5. Compute rolling quarterly (63 trading days) change in repo rate
    macro_df["repo_change_63d"] = macro_df["repo_rate"].diff(63).fillna(0.0)

    # 6. Compute daily risk-free rate for the cash leg (252 trading days/year)
    macro_df["risk_free_daily"] = macro_df["repo_rate"] / 100.0 / 252.0

    # Format Date index back to YYYY-MM-DD
    macro_df.index = macro_df.index.strftime("%Y-%m-%d")
    macro_df.index.name = "Date"

    # 7. Check for missing values
    null_count = macro_df.isnull().sum().sum()
    if null_count > 0:
        raise ValueError(
            f"macro.csv has {null_count} null values! Check date alignment."
        )

    # 8. Save output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    macro_df.to_csv(OUTPUT_PATH)
    print(f"SUCCESS: Saved {len(macro_df)} rows to {OUTPUT_PATH}")
    print(f"Date range: {macro_df.index[0]} to {macro_df.index[-1]}")
    print("\nSample rows:")
    print(macro_df.head())


if __name__ == "__main__":
    build_macro_dataset()