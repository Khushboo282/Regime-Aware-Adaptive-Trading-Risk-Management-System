#Build data/interim/: all series aligned to Nifty 50 trading days, plus daily RBI repo rate.
#python -m src.data              # use existing files in data/raw
#python -m src.data --download   # re-download via yfinance first

import argparse
import json
from datetime import date

import pandas as pd
import yaml

from src.common import INTERIM, RAW, ROOT

EXTERNAL = ROOT / "data" / "external"


def load_config():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)

def groups(cfg):
    core = {**cfg["indian_market"], **cfg["indian_risk_macro"], **cfg["global_drivers"]}
    return {"core": core, "stocks": cfg["stocks"], "sectors": cfg["sectors"]}

def download(cfg):
    import yfinance as yf

    RAW.mkdir(parents=True, exist_ok=True)
    failed = []
    for tickers in groups(cfg).values():
        for name, symbol in tickers.items():
            try:
                df = yf.download(symbol, start=cfg["start_date"], end=cfg.get("end_date"), auto_adjust=cfg["auto_adjust"], progress=False)
                if df.empty:
                    raise ValueError("no data")
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.index = pd.to_datetime(df.index).tz_localize(None)
                df.to_csv(RAW / f"{name}.csv", index_label="Date")
                print(f"[ok] {name}: {len(df)} rows")
            except Exception as e:  # keep going if one ticker fails
                failed.append(name)
                print(f"[failed] {name}: {e}")
    meta = {"pulled_on": str(date.today()), "source": "Yahoo Finance via yfinance", "failed": failed}
    with open(RAW / "_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

def read_close(name):
    return pd.read_csv(RAW / f"{name}.csv", index_col="Date", parse_dates=True)["Close"].rename(name)


def build_interim(cfg):
    INTERIM.mkdir(parents=True, exist_ok=True)
    calendar = read_close(cfg["anchor"]).index
    if cfg.get("exclude_today", True):
        # a candle downloaded during market hours is incomplete, so drop the download day
        with open(RAW / "_metadata.json") as f:
            pulled_on = pd.Timestamp(json.load(f)["pulled_on"])
        calendar = calendar[calendar < pulled_on]

    out = {}
    for group, members in groups(cfg).items():
        frame = pd.concat([read_close(n) for n in members if (RAW / f"{n}.csv").exists()], axis=1)
        frame = frame.reindex(calendar).ffill(limit=cfg["ffill_limit"])
        frame.to_csv(INTERIM / f"{group}_close.csv", index_label="Date")
        out[group] = frame
        missing = int(frame.isna().sum().sum())
        print(f"{group:8s}: {frame.shape[0]} days x {frame.shape[1]} series, {missing} missing cells after the {cfg['ffill_limit']}-day fill")

    # the repo rate only changes on policy dates, so carry each rate forward until the next decision
    repo = pd.read_csv(EXTERNAL / "rbi_repo_rate.csv", parse_dates=["effective_date"])
    repo = repo.set_index("effective_date")["repo_rate_pct"].sort_index()
    every_day = pd.date_range(min(repo.index[0], calendar[0]), calendar[-1], freq="D")
    repo_rate = repo.reindex(every_day).ffill().reindex(calendar).rename("repo_rate")
    assert repo_rate.notna().all(), "repo rate missing on some trading days"
    repo_rate.to_frame().to_csv(INTERIM / "repo_rate.csv", index_label="Date")
    print(f"repo    : {len(repo_rate)} days, {repo.size} policy decisions, range {repo_rate.min()}% to {repo_rate.max()}%")
    out["repo"] = repo_rate
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="download every ticker again first")
    args = parser.parse_args()
    cfg = load_config()
    if args.download:
        download(cfg)
    build_interim(cfg)
