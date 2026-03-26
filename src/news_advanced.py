"""Advanced news feature engineering utilities.

Enhancements:
- Daily headline volume
- Keyword-based event detection
- Daily sentiment aggregation
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

try:
	from src.sentiment import batch_analyze_sentiment
	from src.sentiment_data import fetch_news_headlines
except ImportError:
	from sentiment import batch_analyze_sentiment
	from sentiment_data import fetch_news_headlines


EVENT_KEYWORDS: Dict[str, List[str]] = {
	"merger_flag": ["merger", "acquisition", "takeover", "buyout", "m&a", "merge"],
	"risk_flag": [
		"fraud",
		"lawsuit",
		"penalty",
		"default",
		"bankruptcy",
		"downgrade",
		"probe",
		"investigation",
		"scam",
		"debt",
	],
}


def _headline_flags(headline: str) -> Dict[str, int]:
	"""Create binary event flags for one headline using keyword rules."""
	text = str(headline).lower()
	flags: Dict[str, int] = {}
	for flag_name, keywords in EVENT_KEYWORDS.items():
		flags[flag_name] = int(any(keyword in text for keyword in keywords))
	return flags


def _compute_event_signal(
	news_volume: pd.Series,
	sentiment: pd.Series,
	volume_window: int = 7,
	spike_zscore: float = 1.5,
	extreme_sentiment_threshold: float = 0.6,
) -> pd.Series:
	"""Create event signal based on volume spike and extreme sentiment.

	Rules:
	- +1 when news volume is a sudden spike and sentiment is strongly positive.
	- -1 when news volume is a sudden spike and sentiment is strongly negative.
	- 0 otherwise.
	"""
	rolling_mean = news_volume.rolling(window=volume_window, min_periods=3).mean()
	rolling_std = news_volume.rolling(window=volume_window, min_periods=3).std().fillna(0.0)
	volume_z = (news_volume - rolling_mean) / rolling_std.replace(0.0, pd.NA)
	volume_spike = volume_z.fillna(0.0) >= spike_zscore

	positive_event = volume_spike & (sentiment >= extreme_sentiment_threshold)
	negative_event = volume_spike & (sentiment <= -extreme_sentiment_threshold)

	signal = pd.Series(0, index=news_volume.index, dtype="int64")
	signal.loc[positive_event] = 1
	signal.loc[negative_event] = -1
	return signal


def process_advanced_news(news_df: pd.DataFrame) -> pd.DataFrame:
	"""Aggregate advanced daily news features.

	Args:
		news_df: DataFrame with columns ['date', 'headline'].

	Returns:
		DataFrame with columns ['date', 'sentiment', 'news_volume', 'event_flags', 'event_signal'].
		event_flags is a dictionary with daily binary flags (e.g., merger_flag, risk_flag).
	"""
	if news_df is None or news_df.empty:
		return pd.DataFrame(columns=["date", "sentiment", "news_volume", "event_flags", "event_signal"])

	required_cols = {"date", "headline"}
	missing = required_cols - set(news_df.columns)
	if missing:
		raise ValueError(f"Missing required columns in news_df: {sorted(missing)}")

	df = news_df.copy()
	df["date"] = pd.to_datetime(df["date"]).dt.normalize()
	df["headline"] = df["headline"].astype(str)

	# Sentiment per headline, then average by day.
	df["sentiment_score"] = batch_analyze_sentiment(df["headline"].tolist())

	# Event flags per headline.
	per_headline_flags = df["headline"].apply(_headline_flags)
	for flag_name in EVENT_KEYWORDS:
		df[flag_name] = per_headline_flags.apply(lambda flag_dict: int(flag_dict.get(flag_name, 0)))

	grouped = df.groupby("date", as_index=False).agg(
		sentiment=("sentiment_score", "mean"),
		news_volume=("headline", "count"),
		merger_flag=("merger_flag", "max"),
		risk_flag=("risk_flag", "max"),
	)

	grouped["event_flags"] = grouped.apply(
		lambda row: {
			"merger_flag": int(row["merger_flag"]),
			"risk_flag": int(row["risk_flag"]),
		},
		axis=1,
	)
	grouped["event_signal"] = _compute_event_signal(
		news_volume=grouped["news_volume"],
		sentiment=grouped["sentiment"],
	)

	result = grouped[["date", "sentiment", "news_volume", "event_flags", "event_signal"]].copy()
	result = result.sort_values("date").reset_index(drop=True)
	return result


def build_advanced_news_features(
	symbol: str,
	start_date: str,
	end_date: Optional[str] = None,
	api_key: Optional[str] = None,
) -> pd.DataFrame:
	"""Fetch headlines and build advanced daily news features for a symbol."""
	headlines = fetch_news_headlines(
		symbol=symbol,
		start_date=start_date,
		end_date=end_date,
		api_key=api_key,
	)
	return process_advanced_news(headlines)