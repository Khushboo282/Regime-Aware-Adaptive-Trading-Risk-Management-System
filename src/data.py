"""Download, cache, and merge Indian market data for the regime project.

Run from the repo root:
    python -m src.data --check       # test every ticker quickly, write nothing
    python -m src.data               # download missing tickers, then merge
    python -m src.data --force       # re-download everything
    python -m src.data --merge-only  # rebuild merged files from cached CSVs

Outputs (all CSV, git-ignored):
    data/raw/<name>.csv              one file per ticker (full OHLCV, untouched)
    data/interim/merged.csv          Close prices of market + risk + global series (aligned, not cleaned)
    data/interim/sectors_close.csv   Close prices of the six sector indices (start 2010-2011)
    data/interim/stocks_close.csv    Close prices of the large-cap stock basket
    data/raw/_metadata.json          pull date, source, failed tickers

Cleaning, returns, and features go to data/processed/ (made by the notebooks).
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
REGIME_GROUPS = ["indian_market", "indian_risk_macro", "global_drivers"]


def load_config(path=None):
    with open(path or ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def regime_tickers(cfg):
    """name -> symbol for every series that goes into merged.csv."""
    out = {}
    for group in REGIME_GROUPS:
        out.update(cfg.get(group, {}))
    return out


def all_tickers(cfg):
    return {**regime_tickers(cfg), **cfg.get("sectors", {}), **cfg.get("stocks", {})}


def _clean_frame(df):
    # Newer yfinance versions return MultiIndex columns even for one ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    return df.dropna(how="all")


def download_ticker(symbol, start, end=None, auto_adjust=True, attempts=2):
    """Download one ticker and return an OHLCV DataFrame (raises if empty)."""
    last_error = None
    for _ in range(attempts):
        try:
            df = yf.download(symbol, start=start, end=end,
                             auto_adjust=auto_adjust, progress=False)
            df = _clean_frame(df)
            if not df.empty:
                return df
            last_error = "no data returned"
        except Exception as e:  # network hiccup or bad symbol
            last_error = str(e)
    raise ValueError(f"{symbol}: {last_error}")


def check_tickers(cfg):
    """Quick test: does each ticker return recent data? Writes nothing."""
    print(f"{'name':16s} {'symbol':22s} status")
    bad = []
    for name, symbol in all_tickers(cfg).items():
        try:
            df = _clean_frame(yf.download(symbol, period="1mo", progress=False,
                                          auto_adjust=cfg["auto_adjust"]))
            ok = not df.empty
        except Exception:
            ok = False
        print(f"{name:16s} {symbol:22s} {'ok' if ok else 'NO DATA'}")
        if not ok:
            bad.append(name)
    print(f"\n{len(bad)} ticker(s) without data: {bad}" if bad else "\nAll tickers returned data.")
    return bad


def download_all(cfg, force=False):
    """Download every ticker in the config and cache raw CSVs."""
    raw_dir = ROOT / cfg["paths"]["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    failed = []

    for name, symbol in all_tickers(cfg).items():
        path = raw_dir / f"{name}.csv"
        if path.exists() and not force:
            print(f"[cached]  {name} ({symbol})")
            continue
        try:
            df = download_ticker(symbol, cfg["start_date"], cfg.get("end_date"),
                                 cfg["auto_adjust"])
            df.to_csv(path)
            print(f"[ok]      {name} ({symbol}): {len(df)} rows, "
                  f"{df.index.min().date()} to {df.index.max().date()}")
        except Exception as e:  # one bad ticker must not stop the rest
            print(f"[FAILED]  {name} ({symbol}): {e}")
            failed.append(name)

    meta = {"pulled_on": str(date.today()),
            "source": "Yahoo Finance via yfinance",
            "start_date": cfg["start_date"],
            "tickers": all_tickers(cfg),
            "failed": failed}
    with open(raw_dir / "_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    return failed


def _load_close(raw_dir, names):
    series = {}
    for name in names:
        path = raw_dir / f"{name}.csv"
        if not path.exists():
            print(f"[skip] {name}: no raw file")
            continue
        series[name] = pd.read_csv(path, index_col="Date", parse_dates=True)["Close"]
    return series


def merge_close(cfg):
    """Merge Close prices onto the anchor's trading dates (two output files)."""
    raw_dir = ROOT / cfg["paths"]["raw_dir"]
    out_dir = ROOT / cfg["paths"]["interim_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    anchor = cfg["anchor"]
    anchor_path = raw_dir / f"{anchor}.csv"
    if not anchor_path.exists():
        raise RuntimeError(f"Anchor '{anchor}' has no raw data. Run: python -m src.data")
    calendar = pd.read_csv(anchor_path, index_col="Date", parse_dates=True).index
    if cfg.get("exclude_today", True):
        # today's candle may be incomplete if the data was pulled during market hours
        calendar = calendar[calendar < pd.Timestamp(date.today())]

    results = {}
    for label, names, out_name in [
        ("regime dataset", list(regime_tickers(cfg)), "merged.csv"),
        ("sector indices", list(cfg.get("sectors", {})), "sectors_close.csv"),
        ("stock basket", list(cfg.get("stocks", {})), "stocks_close.csv"),
    ]:
        series = _load_close(raw_dir, names)
        if not series:
            continue
        merged = pd.concat(series, axis=1).reindex(calendar)
        merged = merged.ffill(limit=cfg["ffill_limit"])
        merged.to_csv(out_dir / out_name, index_label="Date")
        results[out_name] = merged
        print(f"\n{label}: {merged.shape[0]} rows x {merged.shape[1]} columns "
              f"({merged.index.min().date()} to {merged.index.max().date()}) -> {out_dir / out_name}")

    return results


def coverage_report(cfg):
    """First date, last date, and rows for every cached ticker."""
    raw_dir = ROOT / cfg["paths"]["raw_dir"]
    rows = []
    for name, symbol in all_tickers(cfg).items():
        path = raw_dir / f"{name}.csv"
        if not path.exists():
            rows.append((name, symbol, None, None, 0))
            continue
        idx = pd.read_csv(path, index_col="Date", parse_dates=True).index
        rows.append((name, symbol, idx.min().date(), idx.max().date(), len(idx)))
    report = pd.DataFrame(rows, columns=["name", "symbol", "first_date", "last_date", "rows"])
    print("\nCoverage of cached raw data:")
    print(report.to_string(index=False))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="test tickers, write nothing")
    parser.add_argument("--force", action="store_true", help="re-download all tickers")
    parser.add_argument("--merge-only", action="store_true", help="skip downloading")
    args = parser.parse_args()

    cfg = load_config()
    if args.check:
        check_tickers(cfg)
        return

    failed = []
    if not args.merge_only:
        failed = download_all(cfg, force=args.force)

    missing_required = [n for n in cfg["required"] if n in failed]
    if missing_required:
        sys.exit(f"\nRequired ticker(s) failed: {missing_required}. Fix config.yaml and rerun.")
    if failed:
        print(f"\nSkipped (failed) tickers: {failed}. They will be missing from the merged files.")

    merge_close(cfg)
    coverage_report(cfg)


if __name__ == "__main__":
    main()