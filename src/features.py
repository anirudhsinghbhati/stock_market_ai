# This project is an advanced stock market prediction system
# It uses historical stock data, technical indicators, and news sentiment
# The goal is to predict whether stock price will go UP or DOWN
# Follow clean coding practices with modular, reusable functions
# Use pandas, numpy, sklearn, yfinance, ta, and transformers where needed
# Write production-level Python code with proper function definitions and docstrings

# Create a module for feature engineering on stock data
#
# Requirements:
# - Write functions to calculate technical indicators using pandas and ta library
# - Input: pandas DataFrame with OHLCV data
# - Output: DataFrame with new feature columns added
#
# Features to implement:
# - Simple Moving Average (SMA) for 10 and 50 days
# - Exponential Moving Average (EMA)
# - Relative Strength Index (RSI)
# - MACD (Moving Average Convergence Divergence)
# - Bollinger Bands (upper and lower)
#
# Additional requirements:
# - Handle NaN values after feature creation
# - Normalize or scale features if needed
# - Keep original columns intact
# - Write a main function called add_features(data) that returns enriched dataset
# - Add docstrings explaining each indicator

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from ta.momentum import rsi
from ta.trend import ADXIndicator
from ta.trend import ema_indicator, macd, macd_signal, sma_indicator
from ta.volume import OnBalanceVolumeIndicator
from ta.volume import VolumeWeightedAveragePrice
from ta.volatility import AverageTrueRange
from ta.volatility import BollingerBands

try:
    from src.data_loader import fetch_stock_data
except ImportError:
    from data_loader import fetch_stock_data


REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
PROTECTED_COLUMNS = {"Date", *REQUIRED_COLUMNS}


logger = logging.getLogger(__name__)


def _validate_input(data: pd.DataFrame) -> None:
    """Validate that required OHLCV columns are present and data is not empty."""
    if data is None or data.empty:
        raise ValueError("Input DataFrame is empty.")

    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Input DataFrame is missing columns: {missing}")


def _add_sma_features(df: pd.DataFrame) -> None:
    """Add simple moving averages (SMA_10 and SMA_50) based on Close price."""
    df["SMA_10"] = sma_indicator(close=df["Close"], window=10)
    df["SMA_50"] = sma_indicator(close=df["Close"], window=50)


def _add_ema_feature(df: pd.DataFrame) -> None:
    """Add exponential moving average (EMA_10) based on Close price."""
    df["EMA_10"] = ema_indicator(close=df["Close"], window=10)


def _add_rsi_feature(df: pd.DataFrame) -> None:
    """Add relative strength index (RSI_14) to capture momentum."""
    df["RSI_14"] = rsi(close=df["Close"], window=14)


def _add_macd_features(df: pd.DataFrame) -> None:
    """Add MACD line and signal line for trend-following momentum."""
    df["MACD"] = macd(close=df["Close"])
    df["MACD_SIGNAL"] = macd_signal(close=df["Close"])


def _add_bollinger_features(df: pd.DataFrame) -> None:
    """Add Bollinger upper and lower bands to represent volatility bounds."""
    bb = BollingerBands(close=df["Close"], window=20, window_dev=2)
    df["BB_UPPER"] = bb.bollinger_hband()
    df["BB_LOWER"] = bb.bollinger_lband()


def _add_lag_and_return_features(df: pd.DataFrame) -> None:
    """Add leakage-safe lagged close features and daily percentage return."""
    df["LAG_1"] = df["Close"].shift(1)
    df["LAG_2"] = df["Close"].shift(2)
    df["LAG_3"] = df["Close"].shift(3)
    df["DAILY_RETURN"] = df["Close"].pct_change()


def _add_volatility_features(df: pd.DataFrame) -> None:
    """Add rolling volatility and rolling volume mean features."""
    df["VOLATILITY_5"] = df["Close"].rolling(window=5).std()
    df["VOLATILITY_10"] = df["Close"].rolling(window=10).std()
    df["VOLUME_MEAN_5"] = df["Volume"].rolling(window=5).mean()
    df["VOLUME_MEAN_10"] = df["Volume"].rolling(window=10).mean()


def _add_advanced_technical_features(df: pd.DataFrame) -> None:
    """Add ATR, ADX, OBV, and VWAP for richer trend/volume/volatility signals."""
    atr_indicator = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=14)
    df["ATR_14"] = atr_indicator.average_true_range()

    adx_indicator = ADXIndicator(high=df["High"], low=df["Low"], close=df["Close"], window=14)
    df["ADX_14"] = adx_indicator.adx()

    obv_indicator = OnBalanceVolumeIndicator(close=df["Close"], volume=df["Volume"])
    df["OBV"] = obv_indicator.on_balance_volume()

    vwap_indicator = VolumeWeightedAveragePrice(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        volume=df["Volume"],
        window=14,
    )
    df["VWAP_14"] = vwap_indicator.volume_weighted_average_price()


def _add_price_return_features(df: pd.DataFrame) -> None:
    """Add multi-horizon close-price returns for momentum profiling."""
    df["RETURN_1D"] = df["Close"].pct_change(periods=1)
    df["RETURN_3D"] = df["Close"].pct_change(periods=3)
    df["RETURN_5D"] = df["Close"].pct_change(periods=5)
    df["RETURN_10D"] = df["Close"].pct_change(periods=10)


def _add_accuracy_booster_features(df: pd.DataFrame) -> None:
    """Add extra predictive features for volume, volatility, and trend strength."""
    df["volume_change"] = df["Volume"].pct_change()
    df["volatility"] = df["Close"].rolling(window=14).std()

    # Trend-focused indicators.
    df["EMA_20"] = ema_indicator(close=df["Close"], window=20)
    df["SMA_20"] = sma_indicator(close=df["Close"], window=20)
    df["trend_strength_ema20"] = (df["Close"] / df["EMA_20"]) - 1.0
    df["sma_crossover_10_50"] = df["SMA_10"] - df["SMA_50"]
    df["adx_trend_flag"] = (df["ADX_14"] >= 25.0).astype(int)


def _add_relative_strength_features(df: pd.DataFrame) -> None:
    """Add stock relative-strength versus NIFTY when market context is available."""
    stock_return_1d = df["Close"].pct_change().fillna(0.0)
    if "nifty_close" in df.columns and pd.api.types.is_numeric_dtype(df["nifty_close"]):
        nifty_return_1d = pd.to_numeric(df["nifty_close"], errors="coerce").pct_change().fillna(0.0)
        df["relative_strength_nifty"] = stock_return_1d - nifty_return_1d
    else:
        # Fallback when NIFTY series is unavailable.
        df["relative_strength_nifty"] = stock_return_1d


def _get_scalable_feature_columns(df: pd.DataFrame, technical_columns: List[str]) -> List[str]:
    """Return numeric columns that can be scaled."""
    columns: List[str] = [
        column
        for column in technical_columns
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column])
    ]
    return columns


def _generated_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Return numeric generated columns excluding core OHLCV and Date."""
    return [
        column
        for column in df.columns
        if column not in PROTECTED_COLUMNS and pd.api.types.is_numeric_dtype(df[column])
    ]


def _lag_feature_columns(df: pd.DataFrame, feature_columns: Sequence[str], lag_days: int = 1) -> pd.DataFrame:
    """Shift generated feature columns to ensure strictly historical inputs."""
    if lag_days <= 0:
        return df

    shifted = df.copy()
    cols = [column for column in feature_columns if column in shifted.columns]
    if cols:
        shifted[cols] = shifted[cols].shift(lag_days)
    return shifted


def _handle_missing_values(df: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    """Fill missing values for generated feature columns in a chronology-safe way."""
    cleaned = df.copy()
    if "Date" in cleaned.columns:
        cleaned = cleaned.sort_values("Date")

    cols = [column for column in feature_columns if column in cleaned.columns]
    if cols:
        cleaned[cols] = cleaned[cols].ffill().bfill().fillna(0.0)

    # Keep rows where required OHLCV is available.
    cleaned = cleaned.dropna(subset=[column for column in REQUIRED_COLUMNS if column in cleaned.columns])
    return cleaned.reset_index(drop=True)


def _remove_highly_correlated_features(
    df: pd.DataFrame,
    threshold: float = 0.90,
    protected_columns: Optional[Set[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Drop numeric features with pairwise correlation above threshold."""
    protected = protected_columns or set()
    numeric_cols = [
        column
        for column in df.select_dtypes(include=["number"]).columns
        if column not in protected
    ]
    if len(numeric_cols) < 2:
        return df, []

    corr_matrix = df[numeric_cols].corr().abs()
    upper = corr_matrix.where(~np.tril(np.ones(corr_matrix.shape, dtype=bool)))
    to_drop = [column for column in upper.columns if (upper[column] > threshold).any()]

    reduced = df.drop(columns=to_drop, errors="ignore")
    return reduced, to_drop


def _normalize_importance(values: pd.Series) -> pd.Series:
    """Normalize feature-importance values to [0, 1]."""
    if values.empty:
        return values
    max_value = float(values.max())
    if max_value <= 0.0:
        return pd.Series(np.zeros(len(values), dtype=float), index=values.index)
    return values / max_value


def _compute_shap_importance(model: Any, X: pd.DataFrame) -> Optional[pd.Series]:
    """Compute mean absolute SHAP importance for fitted tree model.

    Returns None if SHAP is unavailable or computation fails.
    """
    try:
        import shap  # type: ignore
    except Exception:
        return None

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            values = np.asarray(shap_values[1] if len(shap_values) > 1 else shap_values[0], dtype=float)
        else:
            values = np.asarray(shap_values, dtype=float)
            if values.ndim == 3:
                values = values[:, :, 1] if values.shape[2] > 1 else values[:, :, 0]

        if values.ndim != 2:
            return None

        importance = np.mean(np.abs(values), axis=0)
        if importance.shape[0] != X.shape[1]:
            return None
        return pd.Series(importance, index=X.columns, dtype=float)
    except Exception:
        return None


def select_top_features(
    data: pd.DataFrame,
    target: pd.Series | str,
    top_n: int = 20,
    corr_threshold: float = 0.90,
    use_shap: bool = False,
    random_state: int = 42,
) -> List[str]:
    """Select top N features using correlation filtering and model importance.

    Steps:
    1) Remove highly correlated numeric features (above corr_threshold).
    2) Fit RandomForest and compute feature importances.
    3) Optionally blend RandomForest and SHAP importance.

    Args:
        data: Input DataFrame containing candidate features and (optionally) target column.
        target: Target series or target column name in data.
        top_n: Number of top features to return.
        corr_threshold: Correlation filter threshold.
        use_shap: Whether to include SHAP importance in ranking.
        random_state: Random seed for reproducibility.

    Returns:
        Ordered list of top feature names.
    """
    if data is None or data.empty:
        raise ValueError("Input data is empty.")
    if top_n <= 0:
        raise ValueError("top_n must be greater than 0.")

    df = data.copy()

    if isinstance(target, str):
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in data.")
        y = pd.Series(df[target]).reset_index(drop=True)
        feature_df = df.drop(columns=[target])
    else:
        y = pd.Series(target).reset_index(drop=True)
        feature_df = df

    if len(feature_df) != len(y):
        raise ValueError("Feature rows and target length must match.")

    numeric_features = feature_df.select_dtypes(include=["number"]).copy()
    if numeric_features.empty:
        raise ValueError("No numeric feature columns available for feature selection.")

    numeric_features = numeric_features.ffill().bfill().fillna(0.0)
    reduced_df, _ = _remove_highly_correlated_features(
        numeric_features,
        threshold=corr_threshold,
        protected_columns=set(),
    )

    if reduced_df.empty:
        raise ValueError("All features were removed by correlation filtering.")

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(reduced_df, y)

    rf_importance = pd.Series(model.feature_importances_, index=reduced_df.columns, dtype=float)
    rf_norm = _normalize_importance(rf_importance)

    final_scores = rf_norm.copy()
    if use_shap:
        shap_importance = _compute_shap_importance(model, reduced_df)
        if shap_importance is not None:
            shap_norm = _normalize_importance(shap_importance)
            final_scores = 0.5 * rf_norm + 0.5 * shap_norm
        else:
            logger.warning("SHAP requested but unavailable or failed; using RandomForest importance only.")

    ranked = final_scores.sort_values(ascending=False)
    selected = ranked.head(min(top_n, len(ranked))).index.tolist()
    return selected


def add_features(
    data: pd.DataFrame,
    scale_features: bool = False,
    symbol: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    remove_correlated: bool = True,
    corr_threshold: float = 0.90,
    feature_lag_days: int = 1,
) -> pd.DataFrame:
    """Add technical indicators to stock OHLCV data.

    Indicators:
        - SMA 10 and SMA 50
        - EMA 10
        - RSI 14
        - MACD and MACD signal
        - Bollinger upper and lower bands

    External/news/sentiment/market/fundamental features are intentionally
    disabled to keep training and inference fast.

    Args:
        data: Input OHLCV DataFrame.
        scale_features: When ``True``, standardize generated numeric feature columns.
        symbol: Stock symbol used for exogenous feature fetches.
        start_date: Start date in YYYY-MM-DD.
        end_date: Optional end date in YYYY-MM-DD.
        remove_correlated: Drop highly correlated numeric features.
        corr_threshold: Correlation threshold for feature removal.

    Returns:
        Enriched DataFrame with original columns preserved.
    """
    _validate_input(data)
    df = data.copy()

    _add_sma_features(df)
    _add_ema_feature(df)
    _add_rsi_feature(df)
    _add_macd_features(df)
    _add_bollinger_features(df)
    _add_lag_and_return_features(df)
    _add_volatility_features(df)
    _add_advanced_technical_features(df)
    _add_price_return_features(df)
    _add_accuracy_booster_features(df)


    _add_relative_strength_features(df)

    feature_cols = _generated_numeric_columns(df)
    df = _lag_feature_columns(df, feature_columns=feature_cols, lag_days=feature_lag_days)
    df = _handle_missing_values(df, feature_columns=feature_cols)

    if remove_correlated:
        df, _ = _remove_highly_correlated_features(
            df,
            threshold=corr_threshold,
            protected_columns=PROTECTED_COLUMNS,
        )
        feature_cols = _generated_numeric_columns(df)

    if scale_features:
        scaler = StandardScaler()
        scalable_cols = [
            column
            for column in feature_cols
            if pd.api.types.is_numeric_dtype(df[column])
        ]
        if scalable_cols:
            df[scalable_cols] = scaler.fit_transform(df[scalable_cols])

    return df


def build_master_feature_pipeline(
    symbol: str,
    start_date: str,
    end_date: Optional[str] = None,
    scale_features: bool = True,
    remove_correlated: bool = True,
    corr_threshold: float = 0.90,
    feature_lag_days: int = 1,
) -> pd.DataFrame:
    """Build a complete end-to-end feature dataset.

    Steps:
    1. Load stock data
    2. Add technical and trend indicators
    3. Add leakage-safe lagged features
    4. Clean and normalize
    5. Return final dataset
    """
    raw = fetch_stock_data(symbol=symbol, start_date=start_date, end_date=end_date)
    final_df = add_features(
        data=raw,
        scale_features=scale_features,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        remove_correlated=remove_correlated,
        corr_threshold=corr_threshold,
        feature_lag_days=feature_lag_days,
    )
    return final_df


