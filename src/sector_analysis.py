"""Sector analysis features based on competitor basket behavior."""

from __future__ import annotations

from typing import Dict, List, Optional, cast

import pandas as pd
import yfinance as yf


COMPETITOR_MAP: Dict[str, List[str]] = {
	"RELIANCE.NS": ["ONGC.NS", "BPCL.NS", "IOC.NS"],
	"TCS.NS": ["INFY.NS", "WIPRO.NS", "HCLTECH.NS"],
	"INFY.NS": ["TCS.NS", "WIPRO.NS", "HCLTECH.NS"],
	"HDFCBANK.NS": ["ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS"],
	"ICICIBANK.NS": ["HDFCBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS"],
}


def _validate_dates(start_date: str, end_date: Optional[str] = None) -> None:
	"""Validate input dates in YYYY-MM-DD format."""
	try:
		pd.to_datetime(start_date, format="%Y-%m-%d", errors="raise")
		if end_date is not None:
			pd.to_datetime(end_date, format="%Y-%m-%d", errors="raise")
	except Exception as exc:
		raise ValueError("Dates must use the format YYYY-MM-DD.") from exc


def _fetch_close_series(symbol: str, start_date: str, end_date: Optional[str]) -> pd.DataFrame:
	"""Fetch close series for one symbol."""
	raw = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=False)
	frame = cast(pd.DataFrame, raw)

	if frame.empty:
		return pd.DataFrame(columns=["Date", f"{symbol}_close"])

	if isinstance(frame.columns, pd.MultiIndex):
		frame.columns = frame.columns.get_level_values(0)

	close_col = "Adj Close" if "Adj Close" in frame.columns else "Close"
	if close_col not in frame.columns:
		return pd.DataFrame(columns=["Date", f"{symbol}_close"])

	close_df = frame[[close_col]].rename(columns={close_col: f"{symbol}_close"}).reset_index()
	close_df["Date"] = pd.to_datetime(close_df["Date"]).dt.normalize()
	return close_df


def get_competitors(symbol: str) -> List[str]:
	"""Return competitor basket for a symbol."""
	return COMPETITOR_MAP.get(symbol, [])


def fetch_sector_features(symbol: str, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
	"""Compute sector features from competitor close series.

	Features:
	- sector_avg_return: mean return across competitor basket
	- stock_sector_corr_20d: rolling 20-day correlation of stock return with sector average return
	- stock_relative_perf: stock_return - sector_avg_return
	"""
	_validate_dates(start_date=start_date, end_date=end_date)

	competitors = get_competitors(symbol)
	if not competitors:
		return pd.DataFrame(
			columns=["Date", "sector_avg_return", "stock_sector_corr_20d", "stock_relative_perf"]
		)

	stock_close = _fetch_close_series(symbol=symbol, start_date=start_date, end_date=end_date)
	if stock_close.empty:
		return pd.DataFrame(
			columns=["Date", "sector_avg_return", "stock_sector_corr_20d", "stock_relative_perf"]
		)

	merged: pd.DataFrame | None = None
	competitor_close_cols: List[str] = []

	for competitor in competitors:
		comp_df = _fetch_close_series(symbol=competitor, start_date=start_date, end_date=end_date)
		if comp_df.empty:
			continue

		comp_col = f"{competitor}_close"
		competitor_close_cols.append(comp_col)
		merged = comp_df if merged is None else merged.merge(comp_df, on="Date", how="outer")

	if merged is None or not competitor_close_cols:
		return pd.DataFrame(
			columns=["Date", "sector_avg_return", "stock_sector_corr_20d", "stock_relative_perf"]
		)

	merged = merged.sort_values("Date").reset_index(drop=True)
	merged[competitor_close_cols] = merged[competitor_close_cols].ffill().bfill()

	comp_returns = merged[competitor_close_cols].pct_change()
	merged["sector_avg_return"] = comp_returns.mean(axis=1)

	stock_returns = stock_close.rename(columns={f"{symbol}_close": "stock_close"}).copy()
	stock_returns = stock_returns.sort_values("Date").reset_index(drop=True)
	stock_returns["stock_return"] = stock_returns["stock_close"].pct_change()

	features = stock_returns[["Date", "stock_return"]].merge(
		merged[["Date", "sector_avg_return"]], on="Date", how="left"
	)
	features = features.sort_values("Date").reset_index(drop=True)

	features["sector_avg_return"] = features["sector_avg_return"].ffill().bfill().fillna(0.0)
	features["stock_return"] = features["stock_return"].fillna(0.0)

	features["stock_sector_corr_20d"] = (
		features["stock_return"].rolling(window=20, min_periods=5).corr(features["sector_avg_return"])
	)
	features["stock_relative_perf"] = features["stock_return"] - features["sector_avg_return"]

	output = features[["Date", "sector_avg_return", "stock_sector_corr_20d", "stock_relative_perf"]].copy()
	output[["stock_sector_corr_20d", "stock_relative_perf"]] = output[
		["stock_sector_corr_20d", "stock_relative_perf"]
	].ffill().bfill().fillna(0.0)
	return output


def merge_sector_features(stock_df: pd.DataFrame, sector_features_df: pd.DataFrame) -> pd.DataFrame:
	"""Merge sector features into main stock dataset by date."""
	if stock_df is None or stock_df.empty:
		raise ValueError("stock_df is empty.")
	if "Date" not in stock_df.columns:
		raise ValueError("stock_df must contain a 'Date' column.")

	base = stock_df.copy()
	base["Date"] = pd.to_datetime(base["Date"]).dt.normalize()

	if sector_features_df is None or sector_features_df.empty:
		base["sector_avg_return"] = 0.0
		base["stock_sector_corr_20d"] = 0.0
		base["stock_relative_perf"] = 0.0
		return base

	sector = sector_features_df.copy()
	if "Date" not in sector.columns:
		raise ValueError("sector_features_df must contain a 'Date' column.")

	sector["Date"] = pd.to_datetime(sector["Date"]).dt.normalize()

	merged = base.merge(
		sector[["Date", "sector_avg_return", "stock_sector_corr_20d", "stock_relative_perf"]],
		on="Date",
		how="left",
	)

	fill_cols = ["sector_avg_return", "stock_sector_corr_20d", "stock_relative_perf"]
	merged = merged.sort_values("Date")
	merged[fill_cols] = merged[fill_cols].ffill().bfill().fillna(0.0)
	return merged


def add_sector_features(stock_df: pd.DataFrame, symbol: str, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
	"""Fetch and merge sector analysis features into stock data."""
	sector_features = fetch_sector_features(symbol=symbol, start_date=start_date, end_date=end_date)
	return merge_sector_features(stock_df=stock_df, sector_features_df=sector_features)