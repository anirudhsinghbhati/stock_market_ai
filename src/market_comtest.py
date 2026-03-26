"""Market context features from index data.

Builds NIFTY and SENSEX context signals and merges them with stock data.
"""

from __future__ import annotations

from typing import Dict, Optional, cast

import pandas as pd
import yfinance as yf


INDEX_TICKERS: Dict[str, str] = {
    "nifty": "^NSEI",
    "sensex": "^BSESN",
}


def _validate_dates(start_date: str, end_date: Optional[str] = None) -> None:
    """Validate input dates in YYYY-MM-DD format."""
    try:
        pd.to_datetime(start_date, format="%Y-%m-%d", errors="raise")
        if end_date is not None:
            pd.to_datetime(end_date, format="%Y-%m-%d", errors="raise")
    except Exception as exc:
        raise ValueError("Dates must use the format YYYY-MM-DD.") from exc


def _build_index_features(index_name: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Create return/trend/volatility features for one index DataFrame."""
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)

    close_col = "Adj Close" if "Adj Close" in frame.columns else "Close"
    if close_col not in frame.columns:
        raise ValueError(f"Close column missing for index: {index_name}")

    df = frame[[close_col]].rename(columns={close_col: "close"}).copy()
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()

    # Daily return, 5-day trend, and 20-day rolling volatility.
    df[f"{index_name}_return"] = df["close"].pct_change()
    df[f"{index_name}_trend_5d"] = df["close"].pct_change(5)
    df[f"{index_name}_volatility"] = df[f"{index_name}_return"].rolling(window=20, min_periods=5).std()

    keep_cols = ["Date", f"{index_name}_return", f"{index_name}_trend_5d", f"{index_name}_volatility"]
    return df[keep_cols]


def fetch_market_context(start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
    """Fetch NIFTY/SENSEX and return engineered market context features by date."""
    _validate_dates(start_date=start_date, end_date=end_date)

    merged: pd.DataFrame | None = None

    for index_name, ticker in INDEX_TICKERS.items():
        raw = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
        frame = cast(pd.DataFrame, raw)
        if frame.empty:
            continue

        features = _build_index_features(index_name=index_name, frame=frame)
        merged = features if merged is None else merged.merge(features, on="Date", how="outer")

    if merged is None or merged.empty:
        columns = ["Date"]
        for index_name in INDEX_TICKERS:
            columns.extend(
                [
                    f"{index_name}_return",
                    f"{index_name}_trend_5d",
                    f"{index_name}_volatility",
                ]
            )
        return pd.DataFrame(columns=columns)

    merged = merged.sort_values("Date").reset_index(drop=True)
    return merged


def merge_market_context(stock_df: pd.DataFrame, market_context_df: pd.DataFrame) -> pd.DataFrame:
    """Merge market context with stock data and forward-fill context gaps."""
    if stock_df is None or stock_df.empty:
        raise ValueError("stock_df is empty.")
    if "Date" not in stock_df.columns:
        raise ValueError("stock_df must contain a 'Date' column.")

    stock = stock_df.copy()
    stock["Date"] = pd.to_datetime(stock["Date"]).dt.normalize()

    if market_context_df is None or market_context_df.empty:
        return stock

    market = market_context_df.copy()
    if "Date" not in market.columns:
        raise ValueError("market_context_df must contain a 'Date' column.")

    market["Date"] = pd.to_datetime(market["Date"]).dt.normalize()

    merged = stock.merge(market, on="Date", how="left")

    context_cols = [
        col for col in market.columns if col != "Date"
    ]
    if context_cols:
        merged = merged.sort_values("Date")
        merged[context_cols] = merged[context_cols].ffill().bfill().fillna(0.0)

    return merged


def add_market_context_feature(stock_df: pd.DataFrame, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
    """Fetch market context and merge it into the stock dataset."""
    market_context = fetch_market_context(start_date=start_date, end_date=end_date)
    return merge_market_context(stock_df=stock_df, market_context_df=market_context)
