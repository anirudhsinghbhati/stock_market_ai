# This project is an advanced stock market prediction system
# It uses historical stock data, technical indicators, and news sentiment
# The goal is to predict whether stock price will go UP or DOWN
# Follow clean coding practices with modular, reusable functions
# Use pandas, numpy, sklearn, yfinance, ta, and transformers where needed
# Write production-level Python code with proper function definitions and docstrings

from __future__ import annotations

import os
import sys
from datetime import date
from typing import Any

import pandas as pd
import streamlit as st


# Allow app execution with: streamlit run app/streamlit_app.py
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.train import train_stock_model


def _get_feature_importance_df(model: Any, feature_columns: list[str]) -> pd.DataFrame:
    """Return ranked feature importance DataFrame when underlying model supports it."""
    if isinstance(model, dict) and model.get("type") == "ensemble" and "models" in model:
        collected = []
        for member in model["models"].values():
            estimator = member
            if hasattr(member, "named_steps") and "classifier" in member.named_steps:
                estimator = member.named_steps["classifier"]
            if hasattr(estimator, "feature_importances_"):
                values = list(getattr(estimator, "feature_importances_"))
                if len(values) == len(feature_columns):
                    collected.append(values)

        if not collected:
            return pd.DataFrame(columns=["feature", "importance"])

        mean_values = pd.DataFrame(collected).mean(axis=0).tolist()
        importance_df = pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": mean_values,
            }
        ).sort_values("importance", ascending=False)
        return importance_df.reset_index(drop=True)

    estimator = model
    if not isinstance(model, dict) and hasattr(model, "named_steps") and "classifier" in model.named_steps:
        estimator = model.named_steps["classifier"]

    if not hasattr(estimator, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])

    importance_values = list(getattr(estimator, "feature_importances_"))
    if len(importance_values) != len(feature_columns):
        return pd.DataFrame(columns=["feature", "importance"])

    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": importance_values,
        }
    ).sort_values("importance", ascending=False)
    return importance_df.reset_index(drop=True)


st.set_page_config(page_title="Stock Direction Predictor", page_icon="📈", layout="wide")
st.title("Stock Market Direction Predictor")
st.caption("Predict next-day direction (UP or DOWN) from historical prices, technical indicators, and news sentiment.")

with st.sidebar:
    st.header("Configuration")
    symbol = st.text_input("Ticker Symbol", value="RELIANCE.NS").strip()
    start = st.date_input("Start Date", value=date(2020, 1, 1))
    end = st.date_input("End Date", value=date.today())

    st.markdown("### Strategy Parameters")
    model_family = st.selectbox(
        "Model Family",
        options=["ensemble", "informer", "tft"],
        index=0,
        help="Choose classical ensemble or transformer-based time-series model.",
    )
    buy_threshold = st.slider("BUY Threshold (prob_up)", min_value=0.50, max_value=0.95, value=0.70, step=0.01)
    sell_threshold = st.slider("SELL Threshold (prob_up)", min_value=0.05, max_value=0.50, value=0.30, step=0.01)
    sl_stop = st.slider("Stop-Loss (%)", min_value=0.5, max_value=10.0, value=2.0, step=0.1) / 100.0
    tp_stop = st.slider("Take-Profit (%)", min_value=1.0, max_value=20.0, value=4.0, step=0.1) / 100.0

    run_btn = st.button("Train and Predict", type="primary")

if run_btn:
    if not symbol:
        st.error("Please enter a ticker symbol.")
    elif start >= end:
        st.error("Start date must be earlier than end date.")
    else:
        with st.spinner("Fetching data, engineering features, and training model..."):
            try:
                result = train_stock_model(
                    symbol=symbol,
                    start_date=start.strftime("%Y-%m-%d"),
                    end_date=end.strftime("%Y-%m-%d"),
                    buy_threshold=buy_threshold,
                    sell_threshold=sell_threshold,
                    sl_stop=sl_stop,
                    tp_stop=tp_stop,
                    model_family=model_family,
                )
            except Exception as exc:
                st.exception(exc)
            else:
                metrics = result["metrics"]
                dataset = result["dataset"]
                feature_columns = result["feature_columns"]
                prob_up = float(result.get("latest_probability_increase", 0.5))
                expected_ret = float(result.get("latest_expected_return", 0.0))
                regime_label = str(result.get("latest_multiclass_label", "Neutral"))
                trade_signal = str(result.get("trade_signal", "HOLD"))
                stop_loss = result.get("stop_loss")
                take_profit = result.get("take_profit")
                bt_vectorbt = result.get("backtest_vectorbt", {})
                bt_backtrader = result.get("backtest_backtrader", {})
                bt_custom = result.get("backtest_custom", {})
                bt_custom_metrics = bt_custom.get("metrics", {}) if isinstance(bt_custom, dict) else {}

                col1, col2, col3 = st.columns(3)
                col1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                col2.metric("Precision", f"{metrics['precision']:.3f}")
                col3.metric("Recall", f"{metrics['recall']:.3f}")

                advanced_col1, advanced_col2, advanced_col3 = st.columns(3)
                advanced_col1.metric("Probability of Increase", f"{prob_up:.1%}")
                advanced_col2.metric("Expected Return (1D)", f"{expected_ret:.2%}")
                advanced_col3.metric("Market Regime", regime_label)

                st.markdown("### Trade Signal")
                signal_col1, signal_col2, signal_col3 = st.columns(3)
                signal_col1.metric("Action", trade_signal)
                signal_col2.metric(
                    "Stop-Loss",
                    f"{float(stop_loss):.2f}" if stop_loss is not None else "-",
                )
                signal_col3.metric(
                    "Take-Profit",
                    f"{float(take_profit):.2f}" if take_profit is not None else "-",
                )
                st.caption("Rule: BUY if prob_up > 0.70, SELL if prob_up < 0.30, otherwise HOLD.")

                st.markdown("### Backtesting Metrics")
                bt_col1, bt_col2, bt_col3 = st.columns(3)
                bt_col1.metric("VectorBT Sharpe", f"{float(bt_vectorbt.get('sharpe_ratio', 0.0)):.3f}")
                bt_col2.metric("VectorBT Max Drawdown", f"{float(bt_vectorbt.get('max_drawdown', 0.0)):.2%}")
                bt_col3.metric("VectorBT Profit %", f"{float(bt_vectorbt.get('profit_pct', 0.0)):.2f}%")

                bt2_col1, bt2_col2, bt2_col3 = st.columns(3)
                bt2_col1.metric("Backtrader Sharpe", f"{float(bt_backtrader.get('sharpe_ratio', 0.0)):.3f}")
                bt2_col2.metric("Backtrader Max Drawdown", f"{float(bt_backtrader.get('max_drawdown', 0.0)):.2%}")
                bt2_col3.metric("Backtrader Profit %", f"{float(bt_backtrader.get('profit_pct', 0.0)):.2f}%")

                st.markdown("### Equity Curve")
                equity_records = bt_vectorbt.get("equity_curve", [])
                if equity_records:
                    equity_df = pd.DataFrame(equity_records)
                    if "step" in equity_df.columns and "equity" in equity_df.columns:
                        st.line_chart(equity_df.set_index("step")["equity"])
                    else:
                        st.info("Equity curve data format is unavailable.")
                else:
                    st.info("No equity curve data available.")

                st.markdown("### Trade Log")
                trade_records = bt_vectorbt.get("trade_log", [])
                if trade_records:
                    trades_df = pd.DataFrame(trade_records)
                    st.dataframe(trades_df, use_container_width=True)
                else:
                    st.info("No trades were generated for current parameters.")

                st.markdown("### Custom Backtesting")
                custom_col1, custom_col2, custom_col3 = st.columns(3)
                custom_col1.metric(
                    "Total Profit (₹)",
                    f"₹{float(bt_custom_metrics.get('total_profit', 0.0)):,.2f}",
                )
                custom_col2.metric(
                    "Win Rate (%)",
                    f"{float(bt_custom_metrics.get('win_rate', 0.0)) * 100.0:.2f}%",
                )
                custom_col3.metric(
                    "Max Drawdown",
                    f"{float(bt_custom_metrics.get('max_drawdown', 0.0)):.2%}",
                )

                st.markdown("#### Equity Curve Over Time")
                custom_equity = bt_custom.get("equity_curve", []) if isinstance(bt_custom, dict) else []
                if custom_equity:
                    custom_equity_series = pd.Series(custom_equity, dtype=float)
                    if "Date" in dataset.columns:
                        timeline = pd.to_datetime(dataset["Date"], errors="coerce")
                        aligned_dates = timeline.tail(len(custom_equity_series)).reset_index(drop=True)
                        equity_df = pd.DataFrame({"Date": aligned_dates, "Equity": custom_equity_series.values})
                        equity_df = equity_df.dropna(subset=["Date"]).set_index("Date")
                        if equity_df.empty:
                            st.line_chart(custom_equity_series)
                        else:
                            st.line_chart(equity_df["Equity"])
                    else:
                        st.line_chart(custom_equity_series)
                else:
                    st.info("No custom equity curve available.")

                st.markdown("#### Trade History")
                custom_trade_history = bt_custom.get("trade_history", []) if isinstance(bt_custom, dict) else []
                if custom_trade_history:
                    custom_trades_df = pd.DataFrame(custom_trade_history)
                    st.dataframe(custom_trades_df, use_container_width=True)
                else:
                    st.info("No custom trade history available.")

                st.markdown("### Multi-Class Forecast")
                st.write(
                    "Class labels: Strong Up, Weak Up, Neutral, Down. "
                    f"Latest class prediction: {regime_label}"
                )

                if "return_mae" in metrics or "multiclass_accuracy" in metrics:
                    eval_col1, eval_col2 = st.columns(2)
                    eval_col1.metric("Expected Return MAE", f"{metrics.get('return_mae', 0.0):.4f}")
                    eval_col2.metric("Multiclass Accuracy", f"{metrics.get('multiclass_accuracy', 0.0):.3f}")

                if "sentiment_score" in dataset.columns:
                    date_col = "Date" if "Date" in dataset.columns else ("date" if "date" in dataset.columns else None)

                    if date_col is not None:
                        latest_row = dataset.sort_values(date_col).tail(1)
                    else:
                        latest_row = dataset.tail(1)

                    latest_sentiment_value = float(latest_row["sentiment_score"].iloc[0])
                    st.metric("Latest Sentiment Score", f"{latest_sentiment_value:.3f}")

                    st.markdown("### Sentiment Trend (Last 30 Days)")
                    if date_col is None:
                        st.info("Sentiment trend is unavailable because no date column was found.")
                    else:
                        sentiment_trend = (
                            dataset[[date_col, "sentiment_score"]]
                            .dropna()
                            .sort_values(date_col)
                            .tail(30)
                            .copy()
                        )
                        sentiment_trend[date_col] = pd.to_datetime(sentiment_trend[date_col])
                        if sentiment_trend.empty:
                            st.info("No sentiment data available for trend chart.")
                        else:
                            st.line_chart(sentiment_trend.set_index(date_col)["sentiment_score"])
                else:
                    st.info("Sentiment score not available for this run.")

                importance_df = _get_feature_importance_df(result["model"], feature_columns)
                st.markdown("### Feature Importance")
                if importance_df.empty:
                    st.info("Feature importance is unavailable for the selected model.")
                else:
                    st.bar_chart(importance_df.head(15).set_index("feature")["importance"])
                    st.dataframe(importance_df, use_container_width=True)

                    sentiment_importance = importance_df[
                        importance_df["feature"].str.startswith("sentiment")
                    ].reset_index(drop=True)
                    st.markdown("### Sentiment Feature Importance")
                    if sentiment_importance.empty:
                        st.info("No sentiment-based features were found in importance ranking.")
                    else:
                        st.dataframe(sentiment_importance, use_container_width=True)

                st.markdown("### Enriched Dataset Preview")
                st.dataframe(dataset.tail(20), use_container_width=True)

                st.markdown("### Feature Columns Used")
                st.write(feature_columns)
else:
    st.info("Set ticker and dates in the sidebar, then click 'Train and Predict'.")

    