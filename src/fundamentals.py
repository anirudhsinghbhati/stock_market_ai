"""Fundamental data utilities.

Fetches selected fundamentals from yfinance and returns a date-indexed DataFrame.
Because most fundamentals from yfinance are point-in-time snapshots (not daily
history), values are propagated across the requested date range.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd
import yfinance as yf


logger = logging.getLogger(__name__)

FUNDAMENTAL_COLUMNS: List[str] = [
	"trailingPE",
	"forwardPE",
	"profitMargins",
	"returnOnEquity",
	"debtToEquity",
	"revenueGrowth",
]


def _empty_fundamentals_frame(start_date: str, end_date: str) -> pd.DataFrame:
	"""Create an empty date-indexed fundamentals frame for the requested window."""
	start = pd.to_datetime(start_date)
	end = pd.to_datetime(end_date)
	if start > end:
		raise ValueError("start_date must be earlier than or equal to end_date")

	index = pd.date_range(start=start, end=end, freq="D", name="Date")
	frame = pd.DataFrame(index=index, columns=FUNDAMENTAL_COLUMNS, dtype="float64")
	return frame


def _extract_fundamentals(info: Dict[str, object]) -> Dict[str, float]:
	"""Extract required fundamental keys from yfinance info payload."""
	def _to_float(value: Any) -> float:
		try:
			return float(value)
		except (TypeError, ValueError):
			return float("nan")

	extracted: Dict[str, float] = {}
	for key in FUNDAMENTAL_COLUMNS:
		raw_value = info.get(key)
		extracted[key] = _to_float(raw_value)
	return extracted


def fetch_fundamentals(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
	"""Fetch fundamentals for a symbol and return a daily date-indexed DataFrame.

	Args:
		symbol: Yahoo Finance ticker symbol (e.g., RELIANCE.NS).
		start_date: Inclusive start date in YYYY-MM-DD format.
		end_date: Inclusive end date in YYYY-MM-DD format.

	Returns:
		DataFrame indexed by Date with columns:
		trailingPE, forwardPE, profitMargins, returnOnEquity,
		debtToEquity, revenueGrowth.

		Since yfinance fundamentals are typically snapshot values, the function
		forward-fills values across the date range and then back-fills leading
		missing rows.
	"""
	base = _empty_fundamentals_frame(start_date=start_date, end_date=end_date)

	try:
		ticker = yf.Ticker(symbol)
		info = ticker.info or {}
	except Exception as exc:
		logger.warning("Failed to fetch fundamentals for %s: %s", symbol, exc)
		return base.fillna(0.0)

	values = _extract_fundamentals(info)

	if base.empty:
		return base

	# Snapshot fundamentals are applied to the range and then filled for safety.
	for column, value in values.items():
		base[column] = value

	base = base.ffill().bfill()
	base = base.fillna(0.0)
	return base


def merge_fundamentals_with_stock_data(
	stock_df: pd.DataFrame,
	fundamentals_df: pd.DataFrame,
) -> pd.DataFrame:
	"""Merge fundamentals into stock data and forward-fill missing values.

	Fundamental data updates less frequently (often quarterly), so this function
	forward-fills fundamental columns after merge to keep daily rows usable.
	"""
	if stock_df is None or stock_df.empty:
		raise ValueError("stock_df is empty.")

	stock = stock_df.copy()
	if "Date" not in stock.columns:
		raise ValueError("stock_df must contain a 'Date' column.")

	stock["Date"] = pd.to_datetime(stock["Date"]).dt.normalize()

	if fundamentals_df is None or fundamentals_df.empty:
		for column in FUNDAMENTAL_COLUMNS:
			stock[column] = 0.0
		return stock

	fundamentals = fundamentals_df.copy()
	if "Date" in fundamentals.columns:
		fundamentals["Date"] = pd.to_datetime(fundamentals["Date"]).dt.normalize()
	else:
		fundamentals.index = pd.to_datetime(fundamentals.index).normalize()
		fundamentals = fundamentals.reset_index().rename(columns={fundamentals.index.name or "index": "Date"})

	for column in FUNDAMENTAL_COLUMNS:
		if column not in fundamentals.columns:
			fundamentals[column] = float("nan")

	merged = stock.merge(
		fundamentals[["Date"] + FUNDAMENTAL_COLUMNS],
		on="Date",
		how="left",
	)

	merged = merged.sort_values("Date")
	merged[FUNDAMENTAL_COLUMNS] = merged[FUNDAMENTAL_COLUMNS].ffill().bfill().fillna(0.0)
	return merged


def add_fundamentals_feature(stock_df: pd.DataFrame, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
	"""Fetch fundamentals for symbol and merge them into stock data."""
	fundamentals_df = fetch_fundamentals(symbol=symbol, start_date=start_date, end_date=end_date)
	return merge_fundamentals_with_stock_data(stock_df=stock_df, fundamentals_df=fundamentals_df)