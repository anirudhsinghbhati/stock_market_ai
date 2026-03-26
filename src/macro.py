"""Macroeconomic feature utilities.

Fetches major macro proxies and merges derived features into stock data.
"""

from __future__ import annotations

from typing import Dict, Optional, cast

import pandas as pd
import yfinance as yf


MACRO_TICKERS: Dict[str, str] = {
	"crude_oil": "CL=F",      # WTI crude oil futures
	"usd_inr": "INR=X",       # USD/INR exchange rate
	"interest_rate": "^TNX",  # 10Y Treasury yield proxy (available on Yahoo)
}


def _validate_dates(start_date: str, end_date: Optional[str] = None) -> None:
	"""Validate input dates in YYYY-MM-DD format."""
	try:
		pd.to_datetime(start_date, format="%Y-%m-%d", errors="raise")
		if end_date is not None:
			pd.to_datetime(end_date, format="%Y-%m-%d", errors="raise")
	except Exception as exc:
		raise ValueError("Dates must use the format YYYY-MM-DD.") from exc


def _fetch_close_series(ticker: str, feature_name: str, start_date: str, end_date: Optional[str]) -> pd.DataFrame:
	"""Fetch close/adj-close series for a macro ticker."""
	raw = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
	frame = cast(pd.DataFrame, raw)

	if frame.empty:
		return pd.DataFrame(columns=["Date", feature_name])

	if isinstance(frame.columns, pd.MultiIndex):
		frame.columns = frame.columns.get_level_values(0)

	close_col = "Adj Close" if "Adj Close" in frame.columns else "Close"
	if close_col not in frame.columns:
		return pd.DataFrame(columns=["Date", feature_name])

	series = frame[[close_col]].rename(columns={close_col: feature_name}).reset_index()
	series["Date"] = pd.to_datetime(series["Date"]).dt.normalize()
	return series


def fetch_macro_features(start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
	"""Fetch macro indicators and compute daily changes and rolling trends.

	Output columns include, for each base macro feature:
	- <name>
	- <name>_daily_change
	- <name>_trend_5d
	"""
	_validate_dates(start_date=start_date, end_date=end_date)

	merged: pd.DataFrame | None = None
	base_cols: list[str] = []

	for feature_name, ticker in MACRO_TICKERS.items():
		series = _fetch_close_series(
			ticker=ticker,
			feature_name=feature_name,
			start_date=start_date,
			end_date=end_date,
		)
		if series.empty:
			continue

		base_cols.append(feature_name)
		merged = series if merged is None else merged.merge(series, on="Date", how="outer")

	if merged is None or not base_cols:
		output_cols = ["Date"]
		for name in MACRO_TICKERS.keys():
			output_cols.extend([name, f"{name}_daily_change", f"{name}_trend_5d"])
		return pd.DataFrame(columns=output_cols)

	merged = merged.sort_values("Date").reset_index(drop=True)
	merged[base_cols] = merged[base_cols].ffill().bfill()

	for name in base_cols:
		merged[f"{name}_daily_change"] = merged[name].pct_change().fillna(0.0)
		merged[f"{name}_trend_5d"] = merged[name].pct_change(5).fillna(0.0)

	return merged


def merge_macro_features(stock_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
	"""Merge macro features with stock dataset by Date and forward-fill gaps."""
	if stock_df is None or stock_df.empty:
		raise ValueError("stock_df is empty.")
	if "Date" not in stock_df.columns:
		raise ValueError("stock_df must contain a 'Date' column.")

	stock = stock_df.copy()
	stock["Date"] = pd.to_datetime(stock["Date"]).dt.normalize()

	if macro_df is None or macro_df.empty:
		return stock
	if "Date" not in macro_df.columns:
		raise ValueError("macro_df must contain a 'Date' column.")

	macro = macro_df.copy()
	macro["Date"] = pd.to_datetime(macro["Date"]).dt.normalize()

	merged = stock.merge(macro, on="Date", how="left")
	macro_cols = [column for column in macro.columns if column != "Date"]
	if macro_cols:
		merged = merged.sort_values("Date")
		merged[macro_cols] = merged[macro_cols].ffill().bfill().fillna(0.0)

	return merged


def add_macro_features(stock_df: pd.DataFrame, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
	"""Fetch macro indicators and merge engineered macro features into stock data."""
	macro_df = fetch_macro_features(start_date=start_date, end_date=end_date)
	return merge_macro_features(stock_df=stock_df, macro_df=macro_df)