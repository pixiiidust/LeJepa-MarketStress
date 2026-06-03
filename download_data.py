"""
Download raw market data for LeJEPA SOXX POC-1 and save to data/.

Run once before the experiment:
    python download_data.py

Outputs:
    data/soxx.csv   — SOXX OHLCV (2010-01-01 to 2022-12-31)
    data/vix.csv    — ^VIX close (2010-01-01 to 2022-12-31)
    data/dgs10.csv  — ^TNX (10-yr Treasury yield, same units as FRED DGS10)

Note: ^TNX (Yahoo Finance) is used in place of FRED DGS10 because fred.stlouisfed.org
is consistently unreachable. Both series report the 10-year Treasury constant maturity
rate in percent; daily-change values are equivalent for feature engineering purposes.
"""
from __future__ import annotations

from pathlib import Path

import yfinance as yf

START = "2010-01-01"
END = "2022-12-31"
DATA_DIR = Path(__file__).parent / "data"


def download_yfinance(ticker: str, path: Path) -> None:
    df = yf.download(ticker, start=START, end=END, progress=False)
    df.columns = df.columns.get_level_values(0)
    df.to_csv(path)
    print(f"[download] {path.name}: {len(df)} rows")


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    download_yfinance("SOXX", DATA_DIR / "soxx.csv")
    download_yfinance("^VIX", DATA_DIR / "vix.csv")
    download_yfinance("^TNX", DATA_DIR / "dgs10.csv")
    print("[download] Done. Run: python run_lejepa_soxx_poc.py")
