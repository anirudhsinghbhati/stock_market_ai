from __future__ import annotations

import os
import sys
import time
from datetime import date
from typing import Any, Dict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import fetch_stock_data
from src.sentiment_data import fetch_news_headlines
from src.train import get_stock_prediction_summary, train_stock_model


TIMEFRAME_MAP = {
    "Intraday": 60,
    "1 Day": 180,
    "1 Week": 365,
    "1 Month": 365 * 2,
    "6 Months": 365 * 5,
}

SIGNAL_COLORS = {
    "BUY": "#16a34a",
    "SELL": "#dc2626",
    "HOLD": "#f59e0b",
}


def _prediction_to_signal(prediction: str) -> str:
    return "BUY" if str(prediction).upper() == "UP" else "SELL"


def _badge(text: str, signal: str) -> str:
    color = SIGNAL_COLORS.get(signal, "#64748b")
    return (
        f"<span style='background:{color};color:#ffffff;padding:6px 10px;"
        "border-radius:999px;font-weight:700;font-size:12px;'>"
        f"{text}</span>"
    )


def _sentiment_label(score: float) -> str:
    if score > 0.15:
        return "Positive"
    if score < -0.15:
        return "Negative"
    return "Neutral"


def _impact_label(score: float) -> str:
    abs_score = abs(score)
    if abs_score >= 0.5:
        return "High"
    if abs_score >= 0.2:
        return "Medium"
    return "Low"


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


st.set_page_config(page_title="AI Stock Analyzer Pro", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
        * { margin: 0; padding: 0; }
        body { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); }
        .header-container {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            padding: 20px 24px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            margin: -80px -40px 0 -40px;
        }
        .header-title {
            font-size: 32px;
            font-weight: 900;
            color: #ffffff;
            margin: 0;
            background: linear-gradient(135deg, #22c55e 0%, #10b981 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .header-subtitle {
            font-size: 14px;
            color: #94a3b8;
            margin-top: 4px;
        }
        .search-box {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 8px;
            padding: 12px 16px;
            color: #ffffff;
            font-size: 14px;
            margin-top: 12px;
        }
        .card {
            background: rgba(30,41,59,0.7);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 20px;
            backdrop-filter: blur(10px);
            margin-bottom: 16px;
        }
        .card-title {
            font-size: 12px;
            font-weight: 600;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }
        .card-value {
            font-size: 28px;
            font-weight: 700;
            color: #f1f5f9;
        }
        .card-change {
            font-size: 14px;
            margin-top: 6px;
        }
        .signal-badge {
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 18px;
            letter-spacing: 1px;
        }
        .signal-buy { background: rgba(34,197,94,0.2); color: #22c55e; border: 1px solid #22c55e; }
        .signal-sell { background: rgba(239,68,68,0.2); color: #ef4444; border: 1px solid #ef4444; }
        .signal-hold { background: rgba(245,158,11,0.2); color: #f59e0b; border: 1px solid #f59e0b; }
        .metric-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-top: 12px;
        }
        .input-label {
            font-size: 13px;
            font-weight: 600;
            color: #cbd5e1;
            margin-bottom: 6px;
            display: block;
        }
        .control-section {
            background: rgba(30,41,59,0.7);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 20px;
            backdrop-filter: blur(10px);
        }
    </style>
    <div class="header-container">
        <h1 class="header-title">📈 AI Stock Analyzer Pro</h1>
        <p class="header-subtitle">Professional trading insights powered by machine learning</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Top control bar
col1, col2, col3, col4, col5, col6 = st.columns([2, 1.3, 1.3, 1.3, 1.3, 1])
with col1:
    symbol = st.text_input("Stock Ticker", value="RELIANCE.NS", label_visibility="collapsed").strip().upper()
with col2:
    timeframe = st.selectbox("Timeframe", ["Intraday", "1 Day", "1 Week", "1 Month", "6 Months"], index=1, label_visibility="collapsed")
with col3:
    investment = st.number_input("Investment (₹)", min_value=1000.0, value=100000.0, step=10000.0, label_visibility="collapsed")
with col4:
    strategy = st.selectbox("Strategy", ["Conservative", "Moderate", "Aggressive"], index=1, label_visibility="collapsed")
with col5:
    fast_mode = st.toggle("Fast Mode", value=True, help="Skips slower components like backtesting.")
with col6:
    analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

if not analyze_btn:
    st.markdown(
        """
        <div style="text-align: center; padding: 60px 20px; color: #94a3b8;">
            <p style="font-size: 18px; margin-bottom: 10px;">👆 Enter stock ticker and click Analyze to get started</p>
            <p style="font-size: 14px;">Real-time AI predictions • Multi-timeframe analysis • Backtest performance</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

if not symbol:
    st.error("Please provide a stock ticker symbol.")
    st.stop()

today = pd.Timestamp.today().normalize()
lookback_days = TIMEFRAME_MAP[timeframe]
start_date = (today - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
end_date = today.strftime("%Y-%m-%d")

timer_placeholder = st.empty()
timer_start = time.time()
timer_stop = False

spinner_text = "Analyzing stock with Fast Mode..." if fast_mode else "Analyzing stock and running backtests..."
with st.spinner(spinner_text):
    try:
        import threading

        def update_timer():
            while not timer_stop:
                elapsed = time.time() - timer_start
                timer_placeholder.metric("Analysis Time", f"{elapsed:.1f}s")
                time.sleep(0.1)

        timer_thread = threading.Thread(target=update_timer, daemon=True)
        timer_thread.start()

        summary = get_stock_prediction_summary(symbol=symbol, capital=float(investment))
        run = train_stock_model(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            buy_threshold=0.75 if strategy == "Conservative" else (0.7 if strategy == "Moderate" else 0.6),
            sell_threshold=0.25 if strategy == "Conservative" else (0.3 if strategy == "Moderate" else 0.4),
            sl_stop=0.015 if strategy == "Conservative" else (0.02 if strategy == "Moderate" else 0.03),
            tp_stop=0.03 if strategy == "Conservative" else (0.04 if strategy == "Moderate" else 0.06),
            model_family="xgboost",
            save_models=False,
            run_backtests=not fast_mode,
        )
        chart_df = fetch_stock_data(symbol=symbol, start_date=start_date, end_date=end_date)
    except Exception as exc:
        timer_stop = True
        st.exception(exc)
        st.stop()

timer_stop = True
elapsed_time = time.time() - timer_start
timer_placeholder.metric("Analysis Time", f"{elapsed_time:.1f}s")
st.success(f"Analysis completed successfully in {elapsed_time:.1f}s")

levels = dict(summary["trade_levels"])
pnl = dict(summary["profit_loss"])
decision = _prediction_to_signal(summary["prediction"])
confidence = float(summary["confidence"])
risk_level = str(summary["risk_level"])
dataset = run.get("dataset", pd.DataFrame())
entry_price = _to_float(levels.get("entry"), 0.0)
target_price = _to_float(levels.get("target"), 0.0)
stop_loss_price = _to_float(levels.get("stop_loss"), 0.0)

# Section 1: Stock Overview
if not chart_df.empty:
    latest_close = float(chart_df["Close"].iloc[-1])
    prev_close = float(chart_df["Close"].iloc[-2]) if len(chart_df) > 1 else latest_close
    delta = latest_close - prev_close
    delta_pct = (delta / prev_close * 100.0) if prev_close else 0.0
    latest_volume = float(chart_df["Volume"].iloc[-1]) if "Volume" in chart_df.columns else 0.0
    market_trend = "Bullish" if delta >= 0 else "Bearish"
    trend_color = "#22c55e" if delta >= 0 else "#ef4444"
    delta_icon = "📈" if delta >= 0 else "📉"
else:
    latest_close, delta, delta_pct, latest_volume, market_trend = 0.0, 0.0, 0.0, 0.0, "Neutral"
    trend_color = "#f59e0b"
    delta_icon = "➡️"

st.markdown(
    f"""
    <div class="metric-row">
      <div class="metric-card">
        <div style="font-size:12px;color:#94a3b8;margin-bottom:4px;">Current Price</div>
        <div style="font-size:24px;font-weight:700;color:#fff;">{latest_close:,.2f}</div>
        <div style="font-size:12px;color:{trend_color};margin-top:4px;">{delta_icon} {delta:,.2f} ({delta_pct:.2f}%)</div>
      </div>
      <div class="metric-card">
        <div style="font-size:12px;color:#94a3b8;margin-bottom:4px;">Price Change</div>
        <div style="font-size:24px;font-weight:700;color:{trend_color};">{delta:,.2f}</div>
        <div style="font-size:12px;color:#cbd5e1;margin-top:4px;">{delta_pct:.2f}%</div>
      </div>
      <div class="metric-card">
        <div style="font-size:12px;color:#94a3b8;margin-bottom:4px;">Volume</div>
        <div style="font-size:24px;font-weight:700;color:#fff;">{latest_volume:,.0f}</div>
        <div style="font-size:12px;color:{trend_color};margin-top:4px;">Trend: {market_trend}</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div style='margin:20px 0;'></div>", unsafe_allow_html=True)

# Section 2: AI Prediction (Primary Signal)
decision_color = SIGNAL_COLORS.get(decision, "#f59e0b")
st.markdown(
    f"""
    <div style="background:linear-gradient(135deg, rgba({"34,197,94" if decision=="BUY" else ("239,68,68" if decision=="SELL" else "245,158,11")}, 0.1), rgba({"34,197,94" if decision=="BUY" else ("239,68,68" if decision=="SELL" else "245,158,11")}, 0.05));border:1px solid {decision_color};border-radius:12px;padding:24px;text-align:center;">
      <div style="font-size:14px;color:#94a3b8;margin-bottom:8px;">AI Signal</div>
      <div style="font-size:48px;font-weight:900;color:{decision_color};margin-bottom:16px;letter-spacing:-1px;">{decision}</div>
      <div style="background:linear-gradient(90deg, #1e293b, #0f172a);border-radius:999px;padding:8px;margin-bottom:12px;">
        <div style="height:8px;background:{decision_color};border-radius:999px;width:{int(max(0, min(100, confidence * 100)))}%;transition:width 0.3s ease;"></div>
      </div>
      <div style="color:#e2e8f0;font-size:13px;margin-bottom:8px;font-weight:500;">Confidence: {confidence:.1%}</div>
      <div style="display:inline-block;background:{decision_color};color:#000;padding:4px 12px;border-radius:4px;font-size:12px;font-weight:600;">Risk: {risk_level}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div style='margin:24px 0;'></div>", unsafe_allow_html=True)

# Section 3: Multi-timeframe
st.markdown("<div style='font-size:14px;color:#94a3b8;margin-bottom:12px;font-weight:600;text-transform:uppercase;letter-spacing:1px;'>Multi-Timeframe Analysis</div>", unsafe_allow_html=True)
timeframes = {
    "Intraday": summary["timeframes"].get("intraday", "N/A"),
    "1 Day": summary["timeframes"].get("1d", "N/A"),
    "1 Week": "N/A",
    "1 Month": summary["timeframes"].get("1m", "N/A"),
    "6 Months": summary["timeframes"].get("6m", "N/A"),
}
tf_cols = st.columns(5)
for col, (label, val) in zip(tf_cols, timeframes.items()):
    sig = "BUY" if str(val).upper() == "UP" else ("SELL" if str(val).upper() == "DOWN" else "HOLD")
    sig_color = "#22c55e" if sig == "BUY" else ("#ef4444" if sig == "SELL" else "#f59e0b")
    with col:
        st.markdown(
            f"""
            <div style="background:linear-gradient(135deg, #1e293b, #0f172a);border:1px solid {sig_color};border-radius:8px;padding:12px;text-align:center;">
              <div style="font-size:11px;color:#94a3b8;margin-bottom:6px;">{label}</div>
              <div style="font-size:18px;font-weight:700;color:{sig_color};">{sig}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<div style='margin:24px 0;'></div>", unsafe_allow_html=True)

# Section 4: Trade setup
st.markdown("<div style='font-size:14px;color:#94a3b8;margin-bottom:12px;font-weight:600;text-transform:uppercase;letter-spacing:1px;'>Trade Setup</div>", unsafe_allow_html=True)
trade_col1, trade_col2, trade_col3 = st.columns(3, gap="small")

with trade_col1:
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg, #1e293b, #0f172a);border:1px solid #2563eb;border-radius:8px;padding:16px;text-align:center;">
          <div style="font-size:11px;color:#94a3b8;margin-bottom:8px;text-transform:uppercase;">Entry Price</div>
          <div style="font-size:28px;font-weight:700;color:#2563eb;">₹{entry_price:,.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with trade_col2:
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg, #1e293b, #0f172a);border:1px solid #22c55e;border-radius:8px;padding:16px;text-align:center;">
          <div style="font-size:11px;color:#94a3b8;margin-bottom:8px;text-transform:uppercase;">Target Price</div>
          <div style="font-size:28px;font-weight:700;color:#22c55e;">₹{target_price:,.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with trade_col3:
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg, #1e293b, #0f172a);border:1px solid #ef4444;border-radius:8px;padding:16px;text-align:center;">
          <div style="font-size:11px;color:#94a3b8;margin-bottom:8px;text-transform:uppercase;">Stop Loss</div>
          <div style="font-size:28px;font-weight:700;color:#ef4444;">₹{stop_loss_price:,.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='margin:24px 0;'></div>", unsafe_allow_html=True)

# Section 5: Profit/loss calculator
st.markdown("<div style='font-size:14px;color:#94a3b8;margin-bottom:12px;font-weight:600;text-transform:uppercase;letter-spacing:1px;'>Profit / Loss Calculator</div>", unsafe_allow_html=True)
max_profit = float(pnl.get("max_profit_rs", 0.0))
max_loss = float(pnl.get("max_loss_rs", 0.0))
profit_pct = float(pnl.get("profit_percent", 0.0))
loss_pct = float(pnl.get("loss_percent", 0.0))
rr_ratio = (max_profit / max_loss) if max_loss > 0 else 0.0

pnl_col1, pnl_col2, pnl_col3 = st.columns(3, gap="small")

pnl_markdown = f"""
<div style="display:flex;gap:12px;margin-bottom:12px;">
  <div style="flex:1;background:linear-gradient(135deg, #1e293b, #0f172a);border:1px solid #22c55e;border-radius:8px;padding:12px;text-align:center;">
    <div style="font-size:11px;color:#94a3b8;margin-bottom:4px;text-transform:uppercase;">Max Profit</div>
    <div style="font-size:20px;font-weight:700;color:#22c55e;">₹{max_profit:,.0f}</div>
    <div style="font-size:11px;color:#86efac;margin-top:2px;">{profit_pct:.2f}%</div>
  </div>
  <div style="flex:1;background:linear-gradient(135deg, #1e293b, #0f172a);border:1px solid #ef4444;border-radius:8px;padding:12px;text-align:center;">
    <div style="font-size:11px;color:#94a3b8;margin-bottom:4px;text-transform:uppercase;">Max Loss</div>
    <div style="font-size:20px;font-weight:700;color:#ef4444;">-₹{max_loss:,.0f}</div>
    <div style="font-size:11px;color:#fca5a5;margin-top:2px;">{loss_pct:.2f}%</div>
  </div>
  <div style="flex:1;background:linear-gradient(135deg, #1e293b, #0f172a);border:1px solid #f59e0b;border-radius:8px;padding:12px;text-align:center;">
    <div style="font-size:11px;color:#94a3b8;margin-bottom:4px;text-transform:uppercase;">R:R Ratio</div>
    <div style="font-size:20px;font-weight:700;color:#f59e0b;">{rr_ratio:.2f}</div>
  </div>
</div>
"""

total = max_profit + max_loss
green_width = (max_profit / total * 100.0) if total > 0 else 50.0
red_width = 100.0 - green_width
pnl_markdown += f"""
<div style="width:100%;height:12px;border-radius:999px;overflow:hidden;background:#0f172a;border:1px solid #1e293b;">
  <div style="display:flex;height:100%;">
    <div style="width:{green_width:.2f}%;background:linear-gradient(90deg, #16a34a, #22c55e);"></div>
    <div style="width:{red_width:.2f}%;background:linear-gradient(90deg, #dc2626, #ef4444);"></div>
  </div>
</div>
"""

st.markdown(pnl_markdown, unsafe_allow_html=True)

st.markdown("<div style='margin:24px 0;'></div>", unsafe_allow_html=True)

# Section 6: Chart
st.markdown("<div style='font-size:14px;color:#94a3b8;margin-bottom:12px;font-weight:600;text-transform:uppercase;letter-spacing:1px;'>Price Chart</div>", unsafe_allow_html=True)
if chart_df.empty:
    st.info("Chart data is unavailable for the selected timeframe.")
else:
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=chart_df["Date"],
                open=chart_df["Open"],
                high=chart_df["High"],
                low=chart_df["Low"],
                close=chart_df["Close"],
                name=symbol,
            )
        ]
    )
    fig.add_hline(y=entry_price, line_color="#2563eb", line_width=1.5, annotation_text="Entry")
    fig.add_hline(y=target_price, line_color="#16a34a", line_width=1.5, annotation_text="Target")
    fig.add_hline(y=stop_loss_price, line_color="#dc2626", line_width=1.5, annotation_text="Stop")
    fig.update_layout(height=460, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("<div style='margin:24px 0;'></div>", unsafe_allow_html=True)

# Section 7: Sentiment and news
st.markdown("<div style='font-size:14px;color:#94a3b8;margin-bottom:12px;font-weight:600;text-transform:uppercase;letter-spacing:1px;'>Sentiment & News</div>", unsafe_allow_html=True)
latest_sentiment = float(dataset["sentiment_score"].iloc[-1]) if not dataset.empty and "sentiment_score" in dataset.columns else 0.0
sentiment_label = _sentiment_label(latest_sentiment)
impact_level = _impact_label(latest_sentiment)
sent_color = "#22c55e" if latest_sentiment > 0.5 else ("#ef4444" if latest_sentiment < 0 else "#f59e0b")

sent_col1, sent_col2, sent_col3 = st.columns(3, gap="small")

with sent_col1:
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg, #1e293b, #0f172a);border:1px solid {sent_color};border-radius:8px;padding:12px;text-align:center;">
          <div style="font-size:11px;color:#94a3b8;margin-bottom:4px;text-transform:uppercase;">Sentiment Score</div>
          <div style="font-size:24px;font-weight:700;color:{sent_color};">{latest_sentiment:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with sent_col2:
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg, #1e293b, #0f172a);border:1px solid {sent_color};border-radius:8px;padding:12px;text-align:center;">
          <div style="font-size:11px;color:#94a3b8;margin-bottom:4px;text-transform:uppercase;">Sentiment</div>
          <div style="font-size:16px;font-weight:700;color:{sent_color};">{sentiment_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with sent_col3:
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg, #1e293b, #0f172a);border:1px solid {sent_color};border-radius:8px;padding:12px;text-align:center;">
          <div style="font-size:11px;color:#94a3b8;margin-bottom:4px;text-transform:uppercase;">News Impact</div>
          <div style="font-size:16px;font-weight:700;color:{sent_color};">{impact_level}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if fast_mode:
    st.info("Fast Mode is ON: live headlines fetch is skipped for quicker analysis.")
else:
    try:
        headlines_df = fetch_news_headlines(symbol=symbol, start_date=start_date, end_date=end_date)
    except Exception:
        headlines_df = pd.DataFrame()

    if not headlines_df.empty:
        st.markdown("Top Headlines")
        top3 = headlines_df.head(3)
        title_col = "headline" if "headline" in top3.columns else ("title" if "title" in top3.columns else None)
        if title_col is None:
            st.info("Headlines format is unsupported.")
        else:
            for headline in top3[title_col].astype(str).tolist():
                st.write(f"- {headline}")
    else:
        st.info("No recent headlines available.")

if fast_mode:
    st.markdown("<div style='margin:24px 0;'></div>", unsafe_allow_html=True)
    st.info("Fast Mode is ON: backtest section is hidden. Turn OFF Fast Mode to view backtest performance.")
else:
    st.markdown("<div style='margin:24px 0;'></div>", unsafe_allow_html=True)

    # Section 8: Backtest performance
    st.markdown("<div style='font-size:14px;color:#94a3b8;margin-bottom:12px;font-weight:600;text-transform:uppercase;letter-spacing:1px;'>Backtest Performance</div>", unsafe_allow_html=True)
    backtest_custom = run.get("backtest_custom", {})
    custom_metrics = backtest_custom.get("metrics", {}) if isinstance(backtest_custom, dict) else {}
    win_rate_val = float(custom_metrics.get('win_rate', 0.0))
    total_profit_val = float(custom_metrics.get('total_profit', 0.0))
    max_dd_val = float(custom_metrics.get('max_drawdown', 0.0))

    bt_markdown = f"""
    <div style="display:flex;gap:12px;margin-bottom:16px;">
      <div style="flex:1;background:linear-gradient(135deg, #1e293b, #0f172a);border:1px solid #22c55e;border-radius:8px;padding:12px;text-align:center;">
        <div style="font-size:11px;color:#94a3b8;margin-bottom:4px;text-transform:uppercase;">Win Rate</div>
        <div style="font-size:24px;font-weight:700;color:#22c55e;">{win_rate_val*100:.1f}%</div>
      </div>
      <div style="flex:1;background:linear-gradient(135deg, #1e293b, #0f172a);border:1px solid #2563eb;border-radius:8px;padding:12px;text-align:center;">
        <div style="font-size:11px;color:#94a3b8;margin-bottom:4px;text-transform:uppercase;">Total Profit</div>
        <div style="font-size:24px;font-weight:700;color:#2563eb;">₹{total_profit_val:,.0f}</div>
      </div>
      <div style="flex:1;background:linear-gradient(135deg, #1e293b, #0f172a);border:1px solid #ef4444;border-radius:8px;padding:12px;text-align:center;">
        <div style="font-size:11px;color:#94a3b8;margin-bottom:4px;text-transform:uppercase;">Max Drawdown</div>
        <div style="font-size:24px;font-weight:700;color:#ef4444;">{max_dd_val:.2%}</div>
      </div>
    </div>
    """
    st.markdown(bt_markdown, unsafe_allow_html=True)

    equity_curve = backtest_custom.get("equity_curve", []) if isinstance(backtest_custom, dict) else []
    if equity_curve:
        st.line_chart(pd.Series(equity_curve, dtype=float))
    else:
        st.info("No equity curve available.")

    trade_history = backtest_custom.get("trade_history", []) if isinstance(backtest_custom, dict) else []
    if trade_history:
        st.dataframe(pd.DataFrame(trade_history), use_container_width=True)
    else:
        st.info("No trade history available.")

st.markdown("<div style='margin:24px 0;'></div>", unsafe_allow_html=True)

# Section 9: Risk warning
st.markdown(
    """
    <div style="background:linear-gradient(135deg, rgba(239,68,68,0.1), rgba(239,68,68,0.05));border:1px solid #ef4444;border-radius:8px;padding:16px;">
    <div style="color:#fca5a5;font-size:13px;font-weight:500;">⚠️ <strong>Risk Disclaimer</strong></div>
      <div style="color:#cbd5e1;font-size:12px;margin-top:6px;line-height:1.6;">This AI prediction is probabilistic and NOT financial advice. Always use stop-loss, manage risk responsibly, and consult a financial advisor before trading.</div>
    </div>
    """,
    unsafe_allow_html=True,
)
