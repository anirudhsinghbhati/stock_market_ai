# This project is an advanced stock market prediction system
# It uses historical stock data, technical indicators, and news sentiment
# The goal is to predict whether stock price will go UP or DOWN
# Follow clean coding practices with modular, reusable functions
# Use pandas, numpy, sklearn, yfinance, ta, and transformers where needed
# Write production-level Python code with proper function definitions and docstrings

# Create a module for news sentiment analysis using transformer models
#
# Requirements:
# - Use Hugging Face transformers library
# - Load a pretrained sentiment analysis model (BERT or FinBERT)
# - Write a function analyze_sentiment(text: str) -> int
#
# Function behavior:
# - Input: news headline or article text
# - Output:
#     +1 for positive
#     0 for neutral
#     -1 for negative
#
# Additional requirements:
# - Handle empty or invalid text
# - Batch processing function for multiple headlines
# - Function to aggregate sentiment scores by date
# - Include proper docstrings and error handling

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List

import pandas as pd
from transformers import pipeline


_sentiment_analyzer: Callable[..., List[Dict[str, Any]]] | None = None


def _get_sentiment_pipeline() -> Callable[..., List[Dict[str, Any]]]:
    """Lazy-load a Hugging Face sentiment model to avoid import-time overhead."""
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        pipeline_factory: Any = pipeline
        _sentiment_analyzer = pipeline_factory(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )
    if _sentiment_analyzer is None:
        raise RuntimeError("Failed to initialize sentiment pipeline.")
    return _sentiment_analyzer


def _to_score(label: str, confidence: float, neutral_threshold: float = 0.60) -> int:
    """Convert model label/confidence to +1, 0, -1."""
    if confidence < neutral_threshold:
        return 0
    if label.upper() == "POSITIVE":
        return 1
    if label.upper() == "NEGATIVE":
        return -1
    return 0


def analyze_sentiment(text: str) -> int:
    """Analyze one text item and return +1 (positive), 0 (neutral), or -1 (negative)."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text must be a non-empty string.")

    result = _get_sentiment_pipeline()(text)[0]
    return _to_score(result.get("label", ""), float(result.get("score", 0.0)))


def batch_analyze_sentiment(texts: List[str]) -> List[int]:
    """Run sentiment analysis for many texts, mapping invalid items to neutral (0)."""
    if texts is None:
        raise ValueError("Input list of texts cannot be None.")

    cleaned_texts: List[str] = []
    valid_indices: List[int] = []
    scores: List[int] = [0] * len(texts)

    for idx, text in enumerate(texts):
        if isinstance(text, str) and text.strip():
            cleaned_texts.append(text.strip())
            valid_indices.append(idx)

    if not cleaned_texts:
        return scores

    results = _get_sentiment_pipeline()(cleaned_texts)
    for source_idx, result in zip(valid_indices, results):
        scores[source_idx] = _to_score(result.get("label", ""), float(result.get("score", 0.0)))

    return scores


def aggregate_sentiment_by_date(sentiments: Iterable[int], dates: Iterable[str]) -> Dict[str, int]:
    """Aggregate sentiment scores per date and return a date-to-score dictionary.

    Args:
        sentiments: Iterable of sentiment scores (-1, 0, or 1).
        dates: Iterable of date strings in "YYYY-MM-DD" format.

    Returns:
        Dictionary mapping date strings to aggregated sentiment scores.

    Raises:
        ValueError: If lengths of sentiments and dates don't match.
    """
    sentiments_list = list(sentiments)
    dates_list = list(dates)

    if len(sentiments_list) != len(dates_list):
        raise ValueError("Length of sentiments and dates must match.")

    if not sentiments_list:
        return {}

    frame = pd.DataFrame({"date": pd.to_datetime(dates_list), "sentiment": sentiments_list})
    grouped = frame.groupby(frame["date"].dt.strftime("%Y-%m-%d"))["sentiment"].sum()
    return {str(date): int(score) for date, score in grouped.to_dict().items()}


def aggregate_daily_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sentiment scores by date from news headlines.

    Input: DataFrame with columns ['date', 'headline']
    Output: DataFrame with columns ['date', 'sentiment_score']

    Steps:
    - Group headlines by date
    - For each date:
        - Run batch sentiment analysis using existing transformer model
        - Convert labels to numeric (-1, 0, 1)
        - Take average sentiment per day
    - Handle missing days by filling with 0
    - Return sorted DataFrame by date

    Args:
        news_df: DataFrame with columns ['date', 'headline'].

    Returns:
        DataFrame with columns ['date', 'sentiment_score'] sorted by date.

    Raises:
        ValueError: If required columns are missing or DataFrame is empty.
    """
    if news_df is None or news_df.empty:
        raise ValueError("Input DataFrame is empty.")

    required_cols = {"date", "headline"}
    missing_cols = required_cols - set(news_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Ensure date column is datetime
    df = news_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Group headlines by date
    grouped = df.groupby(df["date"].dt.strftime("%Y-%m-%d"))

    daily_sentiments = []

    for date_str, group in grouped:
        headlines = group["headline"].tolist()

        # Run batch sentiment analysis
        sentiment_scores = batch_analyze_sentiment(headlines)

        # Calculate average sentiment for the day
        if sentiment_scores:
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        else:
            avg_sentiment = 0.0

        daily_sentiments.append({"date": date_str, "sentiment_score": avg_sentiment})

    if not daily_sentiments:
        return pd.DataFrame(columns=["date", "sentiment_score"])

    result_df = pd.DataFrame(daily_sentiments)
    result_df["date"] = pd.to_datetime(result_df["date"])

    # Fill missing days with 0 sentiment
    date_range = pd.date_range(start=result_df["date"].min(), end=result_df["date"].max(), freq="D")
    complete_df = pd.DataFrame({"date": date_range})
    result_df = complete_df.merge(result_df, on="date", how="left")
    result_df["sentiment_score"] = result_df["sentiment_score"].fillna(0.0)

    # Sort by date and reset index
    result_df = result_df.sort_values("date").reset_index(drop=True)

    return result_df

