# This project is an advanced stock market prediction system
# It uses historical stock data, technical indicators, and news sentiment
# The goal is to predict whether stock price will go UP or DOWN
# Follow clean coding practices with modular, reusable functions
# Use pandas, numpy, sklearn, yfinance, ta, and transformers where needed
# Write production-level Python code with proper function definitions and docstrings

# Create a module for loading stock market data using yfinance
# 
# Requirements:
# - Write a function named fetch_stock_data
# - It should take parameters:
#     symbol (string, e.g., "RELIANCE.NS")
#     start_date (string, format "YYYY-MM-DD")
#     end_date (optional, default None)
# - Use yfinance to download historical stock data
# - Return the data as a pandas DataFrame
# - Ensure the DataFrame contains columns: Open, High, Low, Close, Volume
# - Reset index so Date becomes a column
# - Handle missing data (drop or fill)
# - Add basic logging/print statements for debugging
# - Include proper docstrings and type hints
# - If no data is returned, raise a meaningful error

from __future__ import annotations

from typing import Optional, cast

import pandas as pd
import yfinance as yf


REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
MARKET_SIGNAL_TICKERS = {
    "nifty_close": "^NSEI",
    "sensex_close": "^BSESN",
    "india_vix_close": "^INDIAVIX",
    "banknifty_close": "^NSEBANK",
    "niftyit_close": "^CNXIT",
}


def fetch_stock_data(symbol: str, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
    """Fetch historical OHLCV data for a ticker using yfinance.

    Args:
        symbol: Stock ticker symbol (for example, ``RELIANCE.NS``).
        start_date: Start date in ``YYYY-MM-DD`` format.
        end_date: Optional end date in ``YYYY-MM-DD`` format.

    Returns:
        A DataFrame with columns ``Date``, ``Open``, ``High``, ``Low``, ``Close``, ``Volume``.

    Raises:
        ValueError: If input dates are invalid, required columns are missing, or no rows are returned.
    """
    if not symbol or not isinstance(symbol, str):
        raise ValueError("'symbol' must be a non-empty string.")

    try:
        pd.to_datetime(start_date, format="%Y-%m-%d", errors="raise")
        if end_date is not None:
            pd.to_datetime(end_date, format="%Y-%m-%d", errors="raise")
    except Exception as exc:
        raise ValueError("Dates must use the format YYYY-MM-DD.") from exc

    print(f"Fetching data for {symbol} from {start_date} to {end_date or 'latest'}")
    data = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=False)
    data = cast(pd.DataFrame, data)

    if data.empty:
        raise ValueError(f"No data returned for {symbol} between {start_date} and {end_date or 'latest'}.")

    # yfinance may return MultiIndex columns for some responses.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required OHLCV columns: {missing_columns}")

    data = data.reset_index()
    data = data[["Date", *REQUIRED_COLUMNS]].copy()
    data = data.dropna(subset=REQUIRED_COLUMNS)

    if data.empty:
        raise ValueError("Data became empty after removing rows with missing OHLCV values.")

    print(f"Fetched {len(data)} clean rows for {symbol}.")
    return data


def fetch_external_market_signals(start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
    """Fetch external market signals that influence stock co-movement.

    Signals include broad indices (NIFTY/SENSEX), volatility index (India VIX),
    and sector proxies (Bank Nifty / Nifty IT) using Yahoo Finance tickers.
    """
    try:
        pd.to_datetime(start_date, format="%Y-%m-%d", errors="raise")
        if end_date is not None:
            pd.to_datetime(end_date, format="%Y-%m-%d", errors="raise")
    except Exception as exc:
        raise ValueError("Dates must use the format YYYY-MM-DD.") from exc

    merged: pd.DataFrame | None = None

    for output_name, ticker in MARKET_SIGNAL_TICKERS.items():
        try:
            raw = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
            frame = cast(pd.DataFrame, raw)
            if frame.empty:
                continue

            if isinstance(frame.columns, pd.MultiIndex):
                frame.columns = frame.columns.get_level_values(0)

            close_col = "Adj Close" if "Adj Close" in frame.columns else "Close"
            if close_col not in frame.columns:
                continue

            signal_df = frame[[close_col]].rename(columns={close_col: output_name}).reset_index()
            signal_df["Date"] = pd.to_datetime(signal_df["Date"]).dt.normalize()

            if merged is None:
                merged = signal_df
            else:
                merged = merged.merge(signal_df, on="Date", how="outer")
        except Exception:
            # Individual signal failures should not break the entire training pipeline.
            continue

    if merged is None or merged.empty:
        return pd.DataFrame(columns=["Date", *MARKET_SIGNAL_TICKERS.keys()])

    merged = merged.sort_values("Date").reset_index(drop=True)
    return merged
    
# Example usage:
# df = fetch_stock_data("RELIANCE.NS", "2020-01-01", "2021-01-01")

