# Create a module to fetch financial news headlines for a given stock symbol
# Use a free API (like NewsAPI or Yahoo Finance news if possible)
# The function should:
# - Take symbol, start_date, end_date
# - Return a pandas DataFrame with columns: ['date', 'headline']
# - Ensure date is in datetime format (only date, no time)
# - Handle API errors and empty responses
# - Include basic logging

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import pandas as pd
import requests


logger = logging.getLogger(__name__)


def fetch_news_headlines(
    symbol: str,
    start_date: str,
    end_date: Optional[str] = None,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch financial news headlines for a stock symbol within a date range.

    Args:
        symbol: Stock ticker symbol (e.g., "AAPL", "RELIANCE.NS").
        start_date: Start date in "YYYY-MM-DD" format.
        end_date: Optional end date in "YYYY-MM-DD" format. If None, uses today's date.
        api_key: Optional NewsAPI key. If None, attempts to use free tier or yfinance fallback.

    Returns:
        A DataFrame with columns ['date', 'headline'] where date is in datetime format.

    Raises:
        ValueError: If input dates are invalid or no headlines are found.
    """
    if not symbol or not isinstance(symbol, str):
        raise ValueError("Symbol must be a non-empty string.")

    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"Invalid start_date format. Expected 'YYYY-MM-DD': {e}")

    if end_date is None:
        end_dt = datetime.now()
    else:
        try:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"Invalid end_date format. Expected 'YYYY-MM-DD': {e}")

    if start_dt >= end_dt:
        raise ValueError("start_date must be earlier than end_date.")

    logger.info(f"Fetching headlines for {symbol} from {start_date} to {end_date or 'today'}")

    # Try NewsAPI first if key is provided
    if api_key:
        return _fetch_from_newsapi(symbol, start_date, end_date or datetime.now().strftime("%Y-%m-%d"), api_key)

    # Fallback: use yfinance news
    logger.info("Attempting to fetch news from yfinance fallback...")
    return _fetch_from_yfinance_fallback(symbol, start_date, end_date)


def _fetch_from_newsapi(symbol: str, start_date: str, end_date: str, api_key: str) -> pd.DataFrame:
    """Fetch headlines from NewsAPI.

    Args:
        symbol: Stock ticker symbol.
        start_date: Start date in "YYYY-MM-DD" format.
        end_date: End date in "YYYY-MM-DD" format.
        api_key: NewsAPI API key.

    Returns:
        DataFrame with columns ['date', 'headline'].

    Raises:
        ValueError: If API request fails or no data is returned.
    """
    query = f"{symbol} stock"
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "from": start_date,
        "to": end_date,
        "sortBy": "publishedAt",
        "language": "en",
        "apiKey": api_key,
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("status") != "ok":
            raise ValueError(f"API error: {data.get('message', 'Unknown error')}")

        articles = data.get("articles", [])
        if not articles:
            logger.warning(f"No articles found for {symbol} in the specified date range.")
            return _create_empty_dataframe()

        headlines = []
        for article in articles:
            published_date = article.get("publishedAt", "").split("T")[0]
            headline = article.get("title", "")
            if published_date and headline:
                headlines.append({"date": published_date, "headline": headline})

        if not headlines:
            logger.warning("No valid headlines extracted from API response.")
            return _create_empty_dataframe()

        df = pd.DataFrame(headlines)
        df["date"] = pd.to_datetime(df["date"])
        logger.info(f"Successfully fetched {len(df)} headlines from NewsAPI.")
        return df

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching from NewsAPI: {e}")
        raise ValueError(f"Failed to fetch news from API: {e}")


def _fetch_from_yfinance_fallback(symbol: str, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
    """Fallback method using yfinance news (if available).

    Args:
        symbol: Stock ticker symbol.
        start_date: Start date in "YYYY-MM-DD" format.
        end_date: Optional end date in "YYYY-MM-DD" format.

    Returns:
        DataFrame with columns ['date', 'headline'].
    """
    try:
        import yfinance as yf

        # Clean up symbol for yfinance if needed
        ticker = yf.Ticker(symbol)
        news_data = ticker.news

        if not news_data:
            logger.warning(f"No news data available from yfinance for {symbol}.")
            return _create_empty_dataframe()

        headlines = []
        for article in news_data:
            # Extract date and headline from article dict
            headline = article.get("title", "") or article.get("headline", "")
            timestamp = article.get("providerPublishTime")

            if timestamp and headline:
                # Convert Unix timestamp to date
                article_date = pd.to_datetime(timestamp, unit="s").strftime("%Y-%m-%d")
                headlines.append({"date": article_date, "headline": headline})

        if not headlines:
            logger.warning("No valid headlines extracted from yfinance.")
            return _create_empty_dataframe()

        df = pd.DataFrame(headlines)
        df["date"] = pd.to_datetime(df["date"])

        # Filter by date range if provided
        if start_date:
            start_dt = pd.to_datetime(start_date)
            df = df[df["date"] >= start_dt]

        if end_date:
            end_dt = pd.to_datetime(end_date)
            df = df[df["date"] <= end_dt]

        logger.info(f"Successfully fetched {len(df)} headlines from yfinance fallback.")
        return df

    except Exception as e:
        logger.error(f"Error fetching from yfinance fallback: {e}")
        return _create_empty_dataframe()


def _create_empty_dataframe() -> pd.DataFrame:
    """Create an empty DataFrame with the correct schema."""
    return pd.DataFrame(columns=["date", "headline"])