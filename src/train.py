# This project is an advanced stock market prediction system
# It uses historical stock data, technical indicators, and news sentiment
# The goal is to predict whether stock price will go UP or DOWN
# Follow clean coding practices with modular, reusable functions
# Use pandas, numpy, sklearn, yfinance, ta, and transformers where needed
# Write production-level Python code with proper function definitions and docstrings

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, TypedDict, cast

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error

try:
    from src.backtesting import BacktestResult, Backtester, run_backtest
    from src.data_loader import fetch_stock_data
    from src.features import add_features
    from src.model import predict, predict_probability, train_model
except ImportError:
    from backtesting import BacktestResult, Backtester, run_backtest
    from data_loader import fetch_stock_data
    from features import add_features
    from model import predict, predict_probability, train_model


logger = logging.getLogger(__name__)
SENTIMENT_FEATURE_COLUMNS: List[str] = []
MULTICLASS_LABELS = {
    0: "Down",
    1: "Neutral",
    2: "Weak Up",
    3: "Strong Up",
}
SUPPORTED_TIMEFRAMES = {"intraday", "1d", "1w", "1m", "6m"}
TIMEFRAME_MIN_ROWS = {
    "intraday": 80,
    "1d": 80,
    "1w": 40,
    "1m": 24,
    "6m": 8,
}
DEFAULT_NSE_STOCKS = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "LT.NS",
    "ITC.NS",
]


class TrainResult(TypedDict):
    model: Any
    return_model: Any
    multiclass_model: Any
    metrics: Dict[str, float]
    dataset: pd.DataFrame
    feature_columns: List[str]
    latest_probability_increase: float
    latest_expected_return: float
    latest_multiclass_label: str
    trade_signal: str
    stop_loss: float | None
    take_profit: float | None
    backtest_vectorbt: BacktestResult
    backtest_backtrader: BacktestResult
    backtest_custom: Dict[str, Any]


class SentimentComparisonResult(TypedDict):
    without_sentiment: Dict[str, float]
    with_sentiment: Dict[str, float]
    differences: Dict[str, float]


class TimeframePrediction(TypedDict):
    prediction: str
    probability: float


class TradeLevelsResult(TypedDict):
    entry: float
    stop_loss: float
    target: float


class TradeSimulationResult(TypedDict):
    outcome: str
    days_taken: int
    actual_profit_or_loss: float


class StockPredictionSummary(TypedDict):
    prediction: str
    confidence: float
    timeframes: Dict[str, str]
    trade_levels: TradeLevelsResult
    profit_loss: Dict[str, float]
    risk_level: str


class WalkForwardFoldResult(TypedDict):
    fold: int
    train_start_index: int
    train_end_index: int
    test_start_index: int
    test_end_index: int
    train_start_date: str
    train_end_date: str
    test_start_date: str
    test_end_date: str
    accuracy: float
    avg_probability: float
    num_test_samples: int


class WalkForwardValidationResult(TypedDict):
    folds: List[WalkForwardFoldResult]
    aggregate: Dict[str, float]


class MultiStockTrainResult(TypedDict):
    model: Any
    metrics: Dict[str, float]
    dataset: pd.DataFrame
    feature_columns: List[str]
    selected_features: List[str]
    symbols_used: List[str]
    total_rows: int


def persist_trained_models(
    model: Any,
    return_model: Any,
    multiclass_model: Any,
    output_dir: str = "models",
) -> Dict[str, str]:
    """Persist trained models to disk using joblib.

    Main model is saved as models/model.pkl to match expected usage.
    """
    os.makedirs(output_dir, exist_ok=True)

    main_model_path = os.path.join(output_dir, "model.pkl")
    return_model_path = os.path.join(output_dir, "return_model.pkl")
    multiclass_model_path = os.path.join(output_dir, "multiclass_model.pkl")

    joblib.dump(model, main_model_path)
    joblib.dump(return_model, return_model_path)
    joblib.dump(multiclass_model, multiclass_model_path)

    return {
        "model": main_model_path,
        "return_model": return_model_path,
        "multiclass_model": multiclass_model_path,
    }


def _generate_trade_levels(prob_up: float, current_price: float, atr_14: float | None) -> Tuple[str, float | None, float | None]:
    """Create BUY/SELL/HOLD signal with ATR-aware stop-loss and take-profit.

    Rules:
    - prob_up > 0.7: BUY
    - prob_up < 0.3: SELL
    - else: HOLD
    """
    if current_price <= 0:
        return "HOLD", None, None

    atr_value = float(atr_14) if atr_14 is not None and atr_14 > 0 else current_price * 0.01
    risk_unit = max(atr_value, current_price * 0.01)

    if prob_up > 0.7:
        signal = "BUY"
        stop_loss = current_price - risk_unit
        take_profit = current_price + (2.0 * risk_unit)
        return signal, float(stop_loss), float(take_profit)

    if prob_up < 0.3:
        signal = "SELL"
        stop_loss = current_price + risk_unit
        take_profit = current_price - (2.0 * risk_unit)
        return signal, float(stop_loss), float(take_profit)

    return "HOLD", None, None


def generate_trade_levels(df: pd.DataFrame, prediction: str, probability: float) -> TradeLevelsResult:
    """Generate ATR-based trade levels from latest row.

    Logic:
    - If prediction == "UP":
        entry = current_price
        stop_loss = current_price - (1.5 * ATR)
        target = current_price + (2.0 * ATR)
    - If prediction == "DOWN":
        entry = current_price
        stop_loss = current_price + (1.5 * ATR)
        target = current_price - (2.0 * ATR)

    Args:
        df: DataFrame containing at least 'Close' and 'ATR_14'. Latest row is used.
        prediction: Expected direction ('UP' or 'DOWN').
        probability: Confidence score for the prediction.

    Returns:
        Dictionary with keys: entry, stop_loss, target.
    """
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty.")
    if "Close" not in df.columns or "ATR_14" not in df.columns:
        raise ValueError("DataFrame must contain 'Close' and 'ATR_14' columns.")

    last = df.tail(1)
    current_price = float(last["Close"].iloc[0])
    atr = float(last["ATR_14"].iloc[0])

    if not pd.notna(current_price) or current_price <= 0:
        raise ValueError("Latest Close price must be a positive numeric value.")
    if not pd.notna(atr) or atr <= 0:
        raise ValueError("Latest ATR_14 must be a positive numeric value.")

    pred = str(prediction).strip().upper()
    _ = float(probability)  # Ensures probability is numeric for caller consistency.

    if pred == "UP":
        stop_loss = current_price - (1.5 * atr)
        target = current_price + (2.0 * atr)
    elif pred == "DOWN":
        stop_loss = current_price + (1.5 * atr)
        target = current_price - (2.0 * atr)
    else:
        raise ValueError("prediction must be either 'UP' or 'DOWN'.")

    return {
        "entry": float(current_price),
        "stop_loss": float(stop_loss),
        "target": float(target),
    }


def classify_risk(probability: float, volatility: float, high_volatility_threshold: float = 0.02) -> str:
    """Classify risk level from prediction confidence and volatility.

    Logic:
    - High: probability < 0.55 OR high volatility
    - Medium: 0.55 <= probability <= 0.7
    - Low: probability > 0.7 and low volatility

    Args:
        probability: Model confidence score in [0, 1].
        volatility: Volatility value (for example ATR/Close or return std).
        high_volatility_threshold: Cutoff above which volatility is considered high.

    Returns:
        "High", "Medium", or "Low".
    """
    prob = float(probability)
    vol = abs(float(volatility))

    if prob < 0.0 or prob > 1.0:
        raise ValueError("probability must be in the range [0, 1].")
    if high_volatility_threshold <= 0:
        raise ValueError("high_volatility_threshold must be > 0.")

    is_high_volatility = vol >= float(high_volatility_threshold)

    if prob < 0.55 or is_high_volatility:
        return "High"
    if prob <= 0.7:
        return "Medium"
    return "Low"


def simulate_trade(
    df: pd.DataFrame,
    entry: float,
    stop_loss: float,
    target: float,
    lookahead_days: int = 10,
) -> TradeSimulationResult:
    """Simulate trade outcomes by checking stop-loss/target hits over next N days.

    For each future day:
    - If target is hit first, outcome is ``target_hit``.
    - If stop-loss is hit first, outcome is ``stop_loss_hit``.

    If neither is hit within lookahead window, outcome is assigned based on final
    close versus entry (profit => target_hit, loss => stop_loss_hit).
    """
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty.")
    required_cols = {"High", "Low", "Close"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")
    if lookahead_days <= 0:
        raise ValueError("lookahead_days must be > 0.")

    entry_price = float(entry)
    stop = float(stop_loss)
    tgt = float(target)

    future = df[["High", "Low", "Close"]].head(lookahead_days).reset_index(drop=True)
    if future.empty:
        raise ValueError("No future rows available for trade simulation.")

    high_values = pd.to_numeric(future["High"], errors="coerce").to_numpy(dtype=float)
    low_values = pd.to_numeric(future["Low"], errors="coerce").to_numpy(dtype=float)

    for day_index, (high, low) in enumerate(zip(high_values, low_values), start=1):

        # If both are touched in the same candle, use conservative stop-loss-first assumption.
        if low <= stop and high >= tgt:
            return {
                "outcome": "stop_loss_hit",
                "days_taken": int(day_index),
                "actual_profit_or_loss": float(stop - entry_price),
            }

        if high >= tgt:
            return {
                "outcome": "target_hit",
                "days_taken": int(day_index),
                "actual_profit_or_loss": float(tgt - entry_price),
            }

        if low <= stop:
            return {
                "outcome": "stop_loss_hit",
                "days_taken": int(day_index),
                "actual_profit_or_loss": float(stop - entry_price),
            }

    final_close = float(future["Close"].iloc[-1])
    pnl = float(final_close - entry_price)
    return {
        "outcome": "target_hit" if pnl >= 0 else "stop_loss_hit",
        "days_taken": int(len(future)),
        "actual_profit_or_loss": pnl,
    }


def _generate_trade_levels_configurable(
    prob_up: float,
    current_price: float,
    atr_14: float | None,
    buy_threshold: float,
    sell_threshold: float,
    sl_stop: float,
    tp_stop: float,
) -> Tuple[str, float | None, float | None]:
    """Create trade action and levels using caller-supplied thresholds and risk params."""
    if current_price <= 0:
        return "HOLD", None, None

    atr_value = float(atr_14) if atr_14 is not None and atr_14 > 0 else current_price * 0.01
    risk_unit = max(atr_value, current_price * max(sl_stop, 0.001))

    if prob_up > buy_threshold:
        return "BUY", float(current_price - risk_unit), float(current_price + (tp_stop / max(sl_stop, 1e-6)) * risk_unit)

    if prob_up < sell_threshold:
        return "SELL", float(current_price + risk_unit), float(current_price - (tp_stop / max(sl_stop, 1e-6)) * risk_unit)

    return "HOLD", None, None


def _sanitize_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Sanitize feature matrix to avoid inf/overflow issues in sklearn models."""
    frame = pd.DataFrame(X).copy()
    if frame.empty:
        return frame

    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.replace([np.inf, -np.inf], np.nan)

    for column in frame.columns:
        series = frame[column]
        median = float(series.median()) if pd.notna(series.median()) else 0.0
        frame[column] = series.fillna(median).fillna(0.0)

    frame = frame.clip(lower=-1e12, upper=1e12)
    return frame


def build_training_dataset(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch data, engineer features, merge sentiment, and create binary target."""
    raw = fetch_stock_data(symbol=symbol, start_date=start_date, end_date=end_date)
    featured = add_features(
        raw,
        scale_features=False,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        remove_correlated=True,
        corr_threshold=0.90,
    )
    return create_target_column(featured)


def fetch_all_stocks(stocks: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Build one combined training dataset across multiple stock symbols.

    Each symbol is processed independently through the same feature pipeline,
    then concatenated into a single dataset for pooled training.
    """
    if not stocks:
        raise ValueError("stocks list cannot be empty.")

    all_data: List[pd.DataFrame] = []
    for symbol in stocks:
        try:
            symbol_df = build_training_dataset(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
            )
            if symbol_df.empty:
                logger.warning("Skipping %s because no rows were returned.", symbol)
                continue

            symbol_df = symbol_df.copy()
            symbol_df["Symbol"] = symbol
            all_data.append(symbol_df)
        except Exception as exc:
            logger.warning("Skipping %s due to data/feature error: %s", symbol, exc)

    if not all_data:
        raise ValueError("No valid stock datasets were generated for the provided symbols.")

    combined = pd.concat(all_data, ignore_index=True)
    if "Date" in combined.columns:
        combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce")
        combined = combined.sort_values(["Date", "Symbol"]).reset_index(drop=True)

    return combined


def encode_symbol_feature(data: pd.DataFrame) -> pd.DataFrame:
    """Encode stock identity using one-hot columns from Symbol."""
    if "Symbol" not in data.columns:
        raise ValueError("Data must contain a 'Symbol' column for symbol encoding.")

    encoded = pd.get_dummies(data, columns=["Symbol"], prefix="Symbol", dtype=float)
    return encoded


def train_multi_stock_model(
    start_date: str,
    end_date: str,
    model_family: str = "xgboost",
    stocks: Optional[List[str]] = None,
) -> MultiStockTrainResult:
    """Train one shared model using pooled data from multiple NSE stocks.

    Pipeline:
    1. Fetch + feature-engineer each stock independently.
    2. Concatenate all rows.
    3. Add one-hot Symbol identity features.
    4. Train one ML model across the pooled dataset.
    """
    stock_list = stocks if stocks is not None else DEFAULT_NSE_STOCKS
    combined = fetch_all_stocks(stock_list, start_date=start_date, end_date=end_date)
    encoded = encode_symbol_feature(combined)

    # Symbols can produce sparse columns after concat (for example stock-specific fundamentals).
    # Fill numeric gaps to preserve rows for pooled training.
    encoded_clean = encoded.copy()
    numeric_cols = encoded_clean.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        encoded_clean[numeric_cols] = (
            encoded_clean[numeric_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

    final_dataset, X, y = prepare_training_matrices(encoded_clean, drop_date=False)
    feature_columns = get_feature_columns(final_dataset)
    if not feature_columns:
        raise ValueError("No numeric feature columns found for multi-stock model training.")

    date_series = final_dataset["Date"] if "Date" in final_dataset.columns else None
    X = X[feature_columns]
    model, metrics = train_model(
        X,
        y,
        feature_columns=feature_columns,
        dates=date_series,
        model_family=model_family,
    )
    selected_features = model.get("selected_features", feature_columns) if isinstance(model, dict) else feature_columns

    metrics["num_symbols"] = float(final_dataset.filter(regex=r"^Symbol_").shape[1])
    metrics["num_rows"] = float(len(final_dataset))

    return {
        "model": model,
        "metrics": metrics,
        "dataset": final_dataset,
        "feature_columns": feature_columns,
        "selected_features": selected_features,
        "symbols_used": sorted(stock_list),
        "total_rows": int(len(final_dataset)),
    }


def _resample_ohlcv(data: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample Date/OHLCV data using pandas while preserving trading semantics."""
    if data.empty:
        return data

    frame = data.copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame = frame.set_index("Date").sort_index()

    resampled = frame.resample(rule).agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    )
    resampled = resampled.dropna(subset=["Open", "High", "Low", "Close"]).reset_index()
    return resampled


def _fetch_intraday_data(symbol: str, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
    """Fetch intraday OHLCV data using 15m with 5m fallback."""
    intervals = ["15m", "5m"]
    for interval in intervals:
        try:
            raw = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False,
                auto_adjust=False,
            )
            data = cast(pd.DataFrame, raw)
            if data.empty:
                continue

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            required = ["Open", "High", "Low", "Close", "Volume"]
            missing = [column for column in required if column not in data.columns]
            if missing:
                continue

            intraday_df = data.reset_index()
            if "Datetime" in intraday_df.columns:
                intraday_df = intraday_df.rename(columns={"Datetime": "Date"})
            intraday_df = intraday_df[["Date", *required]].dropna(subset=required)

            if not intraday_df.empty:
                intraday_df["Date"] = pd.to_datetime(intraday_df["Date"]).dt.tz_localize(None)
                return intraday_df
        except Exception:
            continue

    return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])


def _load_dataset_for_timeframe(symbol: str, timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Load timeframe-specific dataset (intraday, daily, weekly, monthly, 6-month)."""
    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe '{timeframe}'. Supported: {sorted(SUPPORTED_TIMEFRAMES)}")

    if timeframe == "intraday":
        end_ts = pd.to_datetime(end_date)
        intraday_start = max(pd.to_datetime(start_date), end_ts - pd.Timedelta(days=59))
        return _fetch_intraday_data(
            symbol=symbol,
            start_date=intraday_start.strftime("%Y-%m-%d"),
            end_date=end_date,
        )

    daily = fetch_stock_data(symbol=symbol, start_date=start_date, end_date=end_date)

    if timeframe == "1d":
        return daily
    if timeframe == "1w":
        return _resample_ohlcv(daily, rule="W")
    if timeframe == "1m":
        return _resample_ohlcv(daily, rule="ME")

    # 6-month long-term view.
    return _resample_ohlcv(daily, rule="6ME")


def multi_timeframe_predict(
    symbol: str,
    start_date: str,
    end_date: str,
    timeframes: List[str],
    model_family: str = "xgboost",
) -> Dict[str, TimeframePrediction]:
    """Train separate models per timeframe and return prediction + probability.

    Args:
        symbol: Yahoo ticker symbol.
        start_date: Start date YYYY-MM-DD.
        end_date: End date YYYY-MM-DD.
        timeframes: Subset of ["intraday", "1d", "1w", "1m", "6m"].
        model_family: Model family used by train_model.

    Returns:
        Dict mapping timeframe -> {"prediction": "UP|DOWN", "probability": float}
    """
    if not timeframes:
        raise ValueError("timeframes cannot be empty.")

    invalid = [timeframe for timeframe in timeframes if timeframe not in SUPPORTED_TIMEFRAMES]
    if invalid:
        raise ValueError(f"Unsupported timeframes: {invalid}. Supported: {sorted(SUPPORTED_TIMEFRAMES)}")

    results: Dict[str, TimeframePrediction] = {}

    for timeframe in timeframes:
        raw = _load_dataset_for_timeframe(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
        )
        min_rows = TIMEFRAME_MIN_ROWS.get(timeframe, 50)
        if raw.empty or len(raw) < min_rows:
            raise ValueError(f"Insufficient data for timeframe '{timeframe}'.")

        timeframe_start = pd.to_datetime(raw["Date"]).min().strftime("%Y-%m-%d")
        timeframe_end = pd.to_datetime(raw["Date"]).max().strftime("%Y-%m-%d")

        featured = add_features(
            raw,
            scale_features=False,
            symbol=symbol,
            start_date=timeframe_start,
            end_date=timeframe_end,
            remove_correlated=True,
            corr_threshold=0.90,
        )
        labeled = create_target_column(featured)
        final_dataset, X, y = prepare_training_matrices(labeled, drop_date=False)

        feature_columns = get_feature_columns(final_dataset)
        if not feature_columns:
            raise ValueError(f"No numeric features available for timeframe '{timeframe}'.")

        dates = final_dataset["Date"] if "Date" in final_dataset.columns else None
        model, _ = train_model(
            X[feature_columns],
            y,
            feature_columns=feature_columns,
            dates=dates,
            model_family=model_family,
        )

        latest_features = final_dataset[feature_columns].tail(1)
        probability = float(predict_probability(model, latest_features)[0])
        prediction = "UP" if probability >= 0.5 else "DOWN"

        results[timeframe] = {
            "prediction": prediction,
            "probability": probability,
        }

    return results


def get_stock_prediction_summary(symbol: str, capital: float) -> StockPredictionSummary:
    """Build final stock prediction summary with levels, P/L projection, and risk.

    Output schema:
    {
      "prediction": "UP/DOWN",
      "confidence": float,
      "timeframes": {
          "intraday": "UP/DOWN",
          "1d": "UP/DOWN",
          "1m": "UP/DOWN",
          "6m": "UP/DOWN"
      },
      "trade_levels": {"entry": ..., "target": ..., "stop_loss": ...},
      "profit_loss": {
          "max_profit_rs": ...,
          "max_loss_rs": ...,
          "profit_percent": ...,
          "loss_percent": ...
      },
      "risk_level": "High/Medium/Low"
    }
    """
    if not symbol or not isinstance(symbol, str):
        raise ValueError("symbol must be a non-empty string.")
    capital_value = float(capital)
    if capital_value <= 0:
        raise ValueError("capital must be a positive value.")

    end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    start_date = (pd.Timestamp.today() - pd.DateOffset(years=5)).strftime("%Y-%m-%d")

    # Base prediction from daily training pipeline.
    base_result = train_stock_model(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        model_family="xgboost",
        save_models=False,
        run_backtests=False,
    )
    confidence = float(base_result.get("latest_probability_increase", 0.5))
    prediction = "UP" if confidence >= 0.5 else "DOWN"

    # Multi-timeframe directional view.
    timeframe_keys = ["intraday", "1d", "1m", "6m"]
    timeframe_predictions: Dict[str, str] = {}
    for timeframe in timeframe_keys:
        try:
            tf_result = multi_timeframe_predict(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timeframes=[timeframe],
                model_family="xgboost",
            )
            timeframe_predictions[timeframe] = tf_result[timeframe]["prediction"]
        except Exception:
            timeframe_predictions[timeframe] = "N/A"

    dataset = base_result["dataset"]
    levels = generate_trade_levels(df=dataset, prediction=prediction, probability=confidence)

    entry = float(levels["entry"])
    stop_loss = float(levels["stop_loss"])
    target = float(levels["target"])
    position_size = capital_value / entry if entry > 0 else 0.0

    max_profit_per_share = abs(target - entry)
    max_loss_per_share = abs(entry - stop_loss)
    max_profit_rs = float(max_profit_per_share * position_size)
    max_loss_rs = float(max_loss_per_share * position_size)

    profit_percent = float((max_profit_rs / capital_value) * 100.0)
    loss_percent = float((max_loss_rs / capital_value) * 100.0)

    latest_row = dataset.tail(1)
    atr_val = float(latest_row["ATR_14"].iloc[0]) if "ATR_14" in latest_row.columns else 0.0
    close_val = float(latest_row["Close"].iloc[0]) if "Close" in latest_row.columns else 0.0
    volatility = (atr_val / close_val) if close_val > 0 else 0.0
    risk_level = classify_risk(probability=confidence, volatility=volatility)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "timeframes": timeframe_predictions,
        "trade_levels": {
            "entry": entry,
            "target": target,
            "stop_loss": stop_loss,
        },
        "profit_loss": {
            "max_profit_rs": max_profit_rs,
            "max_loss_rs": max_loss_rs,
            "profit_percent": profit_percent,
            "loss_percent": loss_percent,
        },
        "risk_level": risk_level,
    }


def add_external_signals(stock_df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """Fast mode: external market signals are disabled to reduce runtime."""
    if stock_df is None or stock_df.empty:
        raise ValueError("Input stock DataFrame is empty.")
    return stock_df.copy()


def add_sentiment_feature(
    stock_df: pd.DataFrame,
    symbol: str,
    start_date: str,
    end_date: str,
    sentiment_lag_days: int = 1,
) -> pd.DataFrame:
    """Fast mode: sentiment/news features are disabled to reduce runtime."""
    if stock_df is None or stock_df.empty:
        raise ValueError("Input stock DataFrame is empty.")
    return stock_df.copy()


def create_target_column(data: pd.DataFrame) -> pd.DataFrame:
    """Create advanced next-day targets for classification and regression."""
    if "Close" not in data.columns:
        raise ValueError("Input data must contain a 'Close' column to create target.")

    dataset = data.copy()

    # Next-day return target for probabilistic/return forecasting tasks.
    dataset["TargetReturn"] = dataset["Close"].shift(-1) / dataset["Close"] - 1.0

    # If next day's closing price is higher than today -> 1, else 0.
    dataset["Target"] = (dataset["TargetReturn"] > 0).astype(int)

    # 4-class target: Down / Neutral / Weak Up / Strong Up.
    dataset["TargetClass"] = 0
    dataset.loc[dataset["TargetReturn"] >= -0.002, "TargetClass"] = 1
    dataset.loc[dataset["TargetReturn"] >= 0.002, "TargetClass"] = 2
    dataset.loc[dataset["TargetReturn"] >= 0.01, "TargetClass"] = 3
    dataset["TargetClass"] = dataset["TargetClass"].astype(int)

    # Drop final row because shift(-1) makes the future label unavailable.
    dataset = dataset.iloc[:-1].reset_index(drop=True)
    return dataset


def prepare_training_matrices(data: pd.DataFrame, drop_date: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Clean dataset, optionally drop Date, and split into feature matrix X and target y."""
    if "Target" not in data.columns:
        raise ValueError("Input data must contain a 'Target' column.")

    cleaned = data.dropna().reset_index(drop=True)

    drop_columns = ["Target", "TargetReturn", "TargetClass"]
    if drop_date and "Date" in cleaned.columns:
        drop_columns.append("Date")

    X = cleaned.drop(columns=drop_columns)
    y = cleaned["Target"].astype(int)
    return cleaned, X, y


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return numeric feature columns, excluding date/label and including sentiment features."""
    drop_cols = {"Date", "Target", "TargetReturn", "TargetClass"}
    leakage_tokens = ("target", "future", "lead", "next")
    numeric_columns = [
        column
        for column in df.columns
        if column not in drop_cols
        and pd.api.types.is_numeric_dtype(df[column])
        and not any(token in str(column).lower() for token in leakage_tokens)
    ]

    # Make sentiment features explicit in the final feature list when available.
    ordered_without_sentiment = [
        column for column in numeric_columns if column not in SENTIMENT_FEATURE_COLUMNS
    ]
    sentiment_present = [
        column for column in SENTIMENT_FEATURE_COLUMNS if column in numeric_columns
    ]
    return ordered_without_sentiment + sentiment_present


def walk_forward_validation(
    data: pd.DataFrame,
    initial_train_size: int,
    test_size: int,
    step_size: Optional[int] = None,
    model_family: str = "xgboost",
) -> WalkForwardValidationResult:
    """Run chunk-based walk-forward validation and aggregate fold results.

    Workflow:
    - Train on past data only.
    - Test on the next chunk.
    - Move forward and repeat.
    """
    if data is None or data.empty:
        raise ValueError("Input DataFrame is empty.")
    if initial_train_size <= 0:
        raise ValueError("initial_train_size must be > 0.")
    if test_size <= 0:
        raise ValueError("test_size must be > 0.")

    actual_step = int(step_size) if step_size is not None else int(test_size)
    if actual_step <= 0:
        raise ValueError("step_size must be > 0 when provided.")

    final_dataset, X, y = prepare_training_matrices(data, drop_date=False)
    feature_columns = get_feature_columns(final_dataset)
    if not feature_columns:
        raise ValueError("No numeric feature columns found for walk-forward validation.")

    total_rows = len(final_dataset)
    if initial_train_size + test_size > total_rows:
        raise ValueError("Not enough rows for initial train window plus test window.")

    date_series: pd.Series | None = None
    if "Date" in final_dataset.columns:
        date_series = pd.to_datetime(final_dataset["Date"], errors="coerce")

    folds: List[WalkForwardFoldResult] = []

    fold_num = 1
    for test_start in range(initial_train_size, total_rows - test_size + 1, actual_step):
        train_end = test_start
        test_end = test_start + test_size

        X_train = X.iloc[:train_end][feature_columns]
        y_train = y.iloc[:train_end]
        X_test = X.iloc[test_start:test_end][feature_columns]
        y_test = y.iloc[test_start:test_end]

        train_dates = date_series.iloc[:train_end] if date_series is not None else None
        model, _ = train_model(
            X_train,
            y_train,
            feature_columns=feature_columns,
            dates=train_dates,
            model_family=model_family,
        )

        y_pred = predict(model, X_test)
        y_prob = predict_probability(model, X_test)
        fold_accuracy = float(accuracy_score(y_test, y_pred)) if len(y_test) > 0 else 0.0
        fold_avg_probability = float(np.mean(y_prob)) if len(y_prob) > 0 else 0.0

        if date_series is not None:
            train_start_date = str(date_series.iloc[0].date())
            train_end_date = str(date_series.iloc[train_end - 1].date())
            test_start_date = str(date_series.iloc[test_start].date())
            test_end_date = str(date_series.iloc[test_end - 1].date())
        else:
            train_start_date = "N/A"
            train_end_date = "N/A"
            test_start_date = "N/A"
            test_end_date = "N/A"

        folds.append(
            {
                "fold": int(fold_num),
                "train_start_index": 0,
                "train_end_index": int(train_end - 1),
                "test_start_index": int(test_start),
                "test_end_index": int(test_end - 1),
                "train_start_date": train_start_date,
                "train_end_date": train_end_date,
                "test_start_date": test_start_date,
                "test_end_date": test_end_date,
                "accuracy": float(fold_accuracy),
                "avg_probability": float(fold_avg_probability),
                "num_test_samples": int(len(y_test)),
            }
        )
        fold_num += 1

    accuracy_values = [float(fold["accuracy"]) for fold in folds]
    aggregate = {
        "num_folds": float(len(folds)),
        "mean_accuracy": float(np.mean(accuracy_values)) if accuracy_values else 0.0,
        "std_accuracy": float(np.std(accuracy_values)) if accuracy_values else 0.0,
    }

    return {
        "folds": folds,
        "aggregate": aggregate,
    }


def train_stock_model(
    symbol: str,
    start_date: str,
    end_date: str,
    buy_threshold: float = 0.7,
    sell_threshold: float = 0.3,
    sl_stop: float = 0.02,
    tp_stop: float = 0.04,
    model_family: str = "xgboost",
    save_models: bool = False,
    run_backtests: bool = False,
) -> TrainResult:
    """Run the full ML pipeline and return model, metrics, data, and feature list."""
    dataset = build_training_dataset(symbol=symbol, start_date=start_date, end_date=end_date)
    final_dataset, X, y = prepare_training_matrices(dataset, drop_date=False)
    feature_columns = get_feature_columns(final_dataset)

    if not feature_columns:
        raise ValueError("No numeric feature columns found for model training.")

    date_series = final_dataset["Date"] if "Date" in final_dataset.columns else None
    X = X[feature_columns]

    model, metrics = train_model(
        X,
        y,
        feature_columns=feature_columns,
        dates=date_series,
        model_family=model_family,
    )
    selected_features = model.get("selected_features", feature_columns) if isinstance(model, dict) else feature_columns

    X_selected = _sanitize_matrix(X[selected_features])
    y_return = final_dataset["TargetReturn"].astype(float)
    y_class = final_dataset["TargetClass"].astype(int)

    # Auxiliary heads for richer outputs beyond UP/DOWN.
    return_model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    return_model.fit(X_selected, y_return)

    multiclass_model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    multiclass_model.fit(X_selected, y_class)

    return_pred = return_model.predict(X_selected)
    class_pred = multiclass_model.predict(X_selected)
    metrics["return_mae"] = float(mean_absolute_error(y_return, return_pred))
    metrics["multiclass_accuracy"] = float(accuracy_score(y_class, class_pred))

    latest_features = _sanitize_matrix(final_dataset[selected_features].tail(1))
    latest_probability_increase = float(predict_probability(model, latest_features)[0])
    latest_expected_return = float(return_model.predict(latest_features)[0])
    latest_class = int(multiclass_model.predict(latest_features)[0])
    latest_multiclass_label = MULTICLASS_LABELS.get(latest_class, "Neutral")

    latest_row = final_dataset.tail(1)
    latest_close = float(latest_row["Close"].iloc[0])
    latest_atr = float(latest_row["ATR_14"].iloc[0]) if "ATR_14" in latest_row.columns else None
    trade_signal, stop_loss, take_profit = _generate_trade_levels_configurable(
        prob_up=latest_probability_increase,
        current_price=latest_close,
        atr_14=latest_atr,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        sl_stop=sl_stop,
        tp_stop=tp_stop,
    )

    if run_backtests:
        probability_series = pd.Series(predict_probability(model, final_dataset[selected_features]))
        backtest_vectorbt = run_backtest(
            data=final_dataset,
            probability_series=probability_series,
            engine="vectorbt",
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            sl_stop=sl_stop,
            tp_stop=tp_stop,
        )
        backtest_backtrader = run_backtest(
            data=final_dataset,
            probability_series=probability_series,
            engine="backtrader",
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            sl_stop=sl_stop,
            tp_stop=tp_stop,
        )

        # Class-based backtest using generated predictions.
        backtest_input = final_dataset.copy()
        backtest_input["prediction"] = np.where(probability_series >= buy_threshold, "UP", "DOWN")
        backtest_input["probability"] = probability_series.astype(float).values
        if "ATR_14" in backtest_input.columns and "ATR" not in backtest_input.columns:
            backtest_input["ATR"] = pd.to_numeric(backtest_input["ATR_14"], errors="coerce")

        backtester = Backtester(
            data=backtest_input,
            initial_capital=100000.0,
            prediction_column="prediction",
            price_column="Close",
            probability_column="probability",
            atr_column="ATR_14" if "ATR_14" in backtest_input.columns else "ATR",
        )
        backtest_custom = backtester.run_backtest()
        custom_metrics = cast(Dict[str, Any], backtest_custom.get("metrics", {}))
        print(f"Total Profit: {float(custom_metrics.get('total_profit', 0.0)):.2f}")
        print(f"Win Rate: {float(custom_metrics.get('win_rate', 0.0)):.2%}")
        print(f"Drawdown: {float(custom_metrics.get('max_drawdown', 0.0)):.2%}")
    else:
        backtest_vectorbt = {"engine": "vectorbt", "metrics": {}, "equity_curve": [], "trade_history": []}
        backtest_backtrader = {"engine": "backtrader", "metrics": {}, "equity_curve": [], "trade_history": []}
        backtest_custom = {"metrics": {}, "equity_curve": [], "trade_history": []}

    if save_models:
        saved_paths = persist_trained_models(
            model=model,
            return_model=return_model,
            multiclass_model=multiclass_model,
            output_dir="models",
        )
        print(f"Saved models: {saved_paths}")

    return {
        "model": model,
        "return_model": return_model,
        "multiclass_model": multiclass_model,
        "metrics": metrics,
        "dataset": final_dataset,
        "feature_columns": selected_features,
        "latest_probability_increase": latest_probability_increase,
        "latest_expected_return": latest_expected_return,
        "latest_multiclass_label": latest_multiclass_label,
        "trade_signal": trade_signal,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "backtest_vectorbt": backtest_vectorbt,
        "backtest_backtrader": backtest_backtrader,
        "backtest_custom": backtest_custom,
    }


def compare_model_with_without_sentiment(
    symbol: str,
    start_date: str,
    end_date: str,
    plot: bool = False,
) -> SentimentComparisonResult:
    """Compare model metrics with and without sentiment-enhanced features.

    Prints metric differences for accuracy, precision, and recall.
    """
    raw = fetch_stock_data(symbol=symbol, start_date=start_date, end_date=end_date)
    featured = add_features(raw)

    # Baseline run: technical features only.
    baseline_dataset = create_target_column(featured)
    baseline_final, baseline_X, baseline_y = prepare_training_matrices(baseline_dataset, drop_date=False)
    baseline_feature_columns = get_feature_columns(baseline_final)
    baseline_dates = baseline_final["Date"] if "Date" in baseline_final.columns else None
    baseline_model, baseline_metrics = train_model(
        baseline_X[baseline_feature_columns],
        baseline_y,
        feature_columns=baseline_feature_columns,
        dates=baseline_dates,
    )

    # Sentiment run: add sentiment features and derived rolling stats.
    with_sentiment = add_sentiment_feature(
        stock_df=featured,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        sentiment_lag_days=1,
    )
    sentiment_dataset = create_target_column(with_sentiment)
    sentiment_final, sentiment_X, sentiment_y = prepare_training_matrices(sentiment_dataset, drop_date=False)
    sentiment_feature_columns = get_feature_columns(sentiment_final)
    sentiment_dates = sentiment_final["Date"] if "Date" in sentiment_final.columns else None
    sentiment_model, sentiment_metrics = train_model(
        sentiment_X[sentiment_feature_columns],
        sentiment_y,
        feature_columns=sentiment_feature_columns,
        dates=sentiment_dates,
    )

    # Keep trained models referenced for potential debugging/readability in logs.
    _ = baseline_model, sentiment_model

    differences = {
        "accuracy_diff": sentiment_metrics["accuracy"] - baseline_metrics["accuracy"],
        "precision_diff": sentiment_metrics["precision"] - baseline_metrics["precision"],
        "recall_diff": sentiment_metrics["recall"] - baseline_metrics["recall"],
    }

    print("Performance Comparison (With Sentiment - Without Sentiment):")
    print(f"Accuracy difference:  {differences['accuracy_diff']:+.4f}")
    print(f"Precision difference: {differences['precision_diff']:+.4f}")
    print(f"Recall difference:    {differences['recall_diff']:+.4f}")

    if plot:
        metric_names = ["accuracy", "precision", "recall"]
        without_vals = [baseline_metrics[name] for name in metric_names]
        with_vals = [sentiment_metrics[name] for name in metric_names]

        x_positions = range(len(metric_names))
        width = 0.35

        plt.figure(figsize=(8, 5))
        plt.bar([x - width / 2 for x in x_positions], without_vals, width=width, label="Without Sentiment")
        plt.bar([x + width / 2 for x in x_positions], with_vals, width=width, label="With Sentiment")
        plt.xticks(list(x_positions), [name.title() for name in metric_names])
        plt.ylim(0.0, 1.0)
        plt.ylabel("Score")
        plt.title("Model Performance Comparison")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        "without_sentiment": baseline_metrics,
        "with_sentiment": sentiment_metrics,
        "differences": differences,
    }


def predict_latest_direction(model, dataset: pd.DataFrame, feature_columns: List[str]) -> int:
    """Predict latest row direction: 1 (UP) or 0 (DOWN)."""
    latest_features = dataset[feature_columns].tail(1)
    latest_prediction = int(predict(model, latest_features)[0])
    return latest_prediction


if __name__ == "__main__":
    preview = fetch_stock_data(symbol="RELIANCE.NS", start_date="2015-01-01")
    print("Preview of RELIANCE.NS data from 2015-01-01:")
    print(preview.head())
    print(f"Dataset shape: {preview.shape}")

    featured_preview = add_features(preview)
    original_columns = set(preview.columns)
    generated_columns = [column for column in featured_preview.columns if column not in original_columns]

    print(f"Generated feature columns: {generated_columns}")
    print("Feature-enriched sample rows:")
    print(featured_preview.head())

    labeled_preview = create_target_column(featured_preview)
    final_preview, X_preview, y_preview = prepare_training_matrices(labeled_preview, drop_date=True)
    print(f"Final dataset shape after target + cleanup: {final_preview.shape}")
    print(f"X shape: {X_preview.shape}")
    print(f"y shape: {y_preview.shape}")

    result = train_stock_model(symbol="RELIANCE.NS", start_date="2020-01-01", end_date="2023-12-31")
    latest = predict_latest_direction(
        result["model"],
        result["dataset"],
        result["feature_columns"],
    )
    print("Training complete.")
    print(f"Metrics: {result['metrics']}")
    print(f"Latest predicted direction: {'UP' if latest == 1 else 'DOWN'}")