"""Download, cache, and merge market data for the regime project.

Run from the repo root:
    python -m src.data              # download missing tickers, then merge
    python -m src.data --force      # re-download everything
    python -m src.data --merge-only # rebuild merged file from cached CSVs
"""
import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=ROOT / "config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def download_ticker(symbol, start, end=None, auto_adjust=True):
    """Download one ticker and return a clean OHLCV DataFrame."""
    df = yf.download(symbol, start=start, end=end,
                     auto_adjust=auto_adjust, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {symbol}")
    # Newer yfinance versions return MultiIndex columns even for one ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    return df


def download_all(cfg, force=False):
    """Download every ticker in the config and cache raw CSVs."""
    raw_dir = ROOT / cfg["paths"]["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    failed = []

    for name, symbol in cfg["tickers"].items():
        path = raw_dir / f"{name}.csv"
        if path.exists() and not force:
            print(f"[cached]  {name} ({symbol})")
            continue
        try:
            df = download_ticker(symbol, cfg["start_date"],
                                 cfg.get("end_date"), cfg["auto_adjust"])
            df.to_csv(path)
            print(f"[ok]      {name} ({symbol}): {len(df)} rows, "
                  f"{df.index.min().date()} to {df.index.max().date()}")
        except Exception as e:  # keep going so one bad ticker doesn't stop the rest
            print(f"[FAILED]  {name} ({symbol}): {e}")
            failed.append(name)

    meta = {"pulled_on": str(date.today()), "source": "Yahoo Finance via yfinance",
            "tickers": cfg["tickers"], "failed": failed}
    with open(raw_dir / "_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    return failed


def merge_close(cfg):
    """Merge Close prices of all tickers onto the base ticker's trading dates."""
    raw_dir = ROOT / cfg["paths"]["raw_dir"]
    out_dir = ROOT / cfg["paths"]["processed_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    series = {}
    for name in cfg["tickers"]:
        path = raw_dir / f"{name}.csv"
        if not path.exists():
            print(f"[skip] {name}: no raw file")
            continue
        df = pd.read_csv(path, index_col="Date", parse_dates=True)
        series[name] = df["Close"]

    base = cfg["base_ticker"]
    if base not in series:
        raise RuntimeError(f"Base ticker '{base}' has no raw data")

    merged = pd.concat(series, axis=1)
    merged = merged.reindex(series[base].index)           # base-ticker trading days
    merged = merged.ffill(limit=cfg["ffill_limit"])      # fill short holiday gaps only

    out_path = out_dir / "merged.csv"
    merged.to_csv(out_path, index_label="Date")

    print(f"\nMerged shape: {merged.shape}")
    print(f"Date range:   {merged.index.min().date()} to {merged.index.max().date()}")
    print("Missing values per column:")
    print(merged.isna().sum().to_string())
    print(f"Saved to {out_path}")
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="re-download all tickers")
    parser.add_argument("--merge-only", action="store_true", help="skip downloading")
    args = parser.parse_args()

    cfg = load_config()
    if not args.merge_only:
        failed = download_all(cfg, force=args.force)
        if failed:
            print(f"\nFailed tickers: {failed}. Fix the symbol in config.yaml and rerun.")
    merge_close(cfg)


if __name__ == "__main__":
    main()