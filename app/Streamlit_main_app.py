from __future__ import annotations

import json
import os
import sys
import base64
from datetime import date, datetime, time
from typing import Any, Dict
from urllib.error import URLError
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import fetch_stock_data
from src.sentiment_data import fetch_news_headlines
from src.train import (
    DEFAULT_NSE_STOCKS,
    classify_risk,
    generate_trade_levels,
    get_stock_prediction_summary,
    train_sector_models,
    train_stock_model,
)

try:
    from src.train import predict_with_saved_sector_router
except ImportError:
    predict_with_saved_sector_router = None

TIMEFRAME_MAP = {
    "Intraday": 60,
    "1 Day": 180,
    "1 Week": 365,
    "1 Month": 365 * 2,
    "6 Months": 365 * 5,
    "1 Year": 365 * 7,
    "2 Years": 365 * 10,
}

SIGNAL_COLORS = {
    "BUY": "#16a34a",
    "SELL": "#dc2626",
    "HOLD": "#f59e0b",
}

# Curated list of popular Indian stocks for quick selection (15+ options)
POPULAR_STOCKS = [
    ("RELIANCE.NS", "Reliance Industries", "Energy"),
    ("TCS.NS", "Tata Consultancy Services", "IT"),
    ("INFY.NS", "Infosys", "IT"),
    ("HDFCBANK.NS", "HDFC Bank", "Banking"),
    ("ICICIBANK.NS", "ICICI Bank", "Banking"),
    ("SBIN.NS", "State Bank of India", "Banking"),
    ("WIPRO.NS", "Wipro", "IT"),
    ("AXISBANK.NS", "Axis Bank", "Banking"),
    ("LT.NS", "Larsen & Toubro", "Infrastructure"),
    ("ITC.NS", "ITC Limited", "FMCG/Diversified"),
    ("ASIANPAINT.NS", "Asian Paints", "FMCG"),
    ("MARUTI.NS", "Maruti Suzuki", "Automobiles"),
    ("HDFC.NS", "HDFC Limited", "Finance"),
    ("SUNPHARMA.NS", "Sun Pharmaceutical", "Pharma"),
    ("BAJAJFINSV.NS", "Bajaj Financial Services", "Finance"),
]

POPULAR_STOCKS_MAP = {symbol: {"name": name, "sector": sector} for symbol, name, sector in POPULAR_STOCKS}
PEER_SYMBOLS_BY_SECTOR = {
    "Banking": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS"],
    "IT": ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"],
    "Energy": ["RELIANCE.NS", "ONGC.NS", "BPCL.NS", "IOC.NS", "HINDPETRO.NS"],
    "FMCG": ["ITC.NS", "HINDUNILVR.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS"],
    "FMCG/Diversified": ["ITC.NS", "HINDUNILVR.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS"],
    "Infrastructure": ["LT.NS", "ULTRACEMCO.NS", "ADANIPORTS.NS", "SIEMENS.NS", "TATAPOWER.NS"],
    "Automobiles": ["MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "EICHERMOT.NS", "BAJAJ-AUTO.NS"],
    "Pharma": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "LUPIN.NS"],
    "Finance": ["BAJAJFINSV.NS", "BAJFINANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS"],
}


def _get_symbol_sector(symbol: str) -> str:
    info = POPULAR_STOCKS_MAP.get(str(symbol or "").upper())
    if info is not None:
        return str(info.get("sector") or "Unknown")
    return "Unknown"


def _format_large_number(value: Any) -> str:
    number = _to_float(value, 0.0)
    abs_number = abs(number)
    if abs_number >= 1_00_00_00_00_000:
        return f"₹{number / 1_00_00_00_00_000:.2f} L Cr"
    if abs_number >= 1_00_00_00_000:
        return f"₹{number / 1_00_00_00_000:.2f} Cr"
    if abs_number >= 1_00_000:
        return f"₹{number / 1_00_000:.2f} L"
    return f"₹{number:,.0f}"


def _is_nse_market_open_now() -> bool:
    """Return True when current IST time is within regular NSE trading hours."""
    now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
    if now_ist.weekday() >= 5:
        return False
    market_open = time(9, 15)
    market_close = time(15, 30)
    return market_open <= now_ist.time() <= market_close


@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_company_profile(symbol: str) -> dict[str, Any]:
    """Fetch company profile and fundamentals from Yahoo Finance with safe fallbacks."""
    ticker = yf.Ticker(symbol)
    profile: dict[str, Any] = {
        "symbol": symbol,
        "name": symbol,
        "sector": _get_symbol_sector(symbol),
        "industry": "N/A",
        "summary": "Company summary not available.",
        "market_cap": None,
        "pe_ratio": None,
        "pb_ratio": None,
        "dividend_yield": None,
        "roe": None,
        "debt_to_equity": None,
        "fifty_two_week_high": None,
        "fifty_two_week_low": None,
    }
    try:
        info = ticker.info if isinstance(ticker.info, dict) else {}
    except Exception:
        info = {}

    profile["name"] = str(info.get("longName") or info.get("shortName") or symbol)
    profile["sector"] = str(info.get("sector") or profile["sector"])
    profile["industry"] = str(info.get("industry") or "N/A")
    profile["summary"] = str(info.get("longBusinessSummary") or profile["summary"])
    profile["market_cap"] = info.get("marketCap")
    profile["pe_ratio"] = info.get("trailingPE")
    profile["pb_ratio"] = info.get("priceToBook")
    profile["dividend_yield"] = info.get("dividendYield")
    profile["roe"] = info.get("returnOnEquity")
    profile["debt_to_equity"] = info.get("debtToEquity")
    profile["fifty_two_week_high"] = info.get("fiftyTwoWeekHigh")
    profile["fifty_two_week_low"] = info.get("fiftyTwoWeekLow")
    return profile


@st.cache_data(ttl=1200, show_spinner=False)
def _build_peer_comparison(symbol: str, start_date: str, end_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build peer table and normalized performance data for non-fast mode insights."""
    sector = _get_symbol_sector(symbol)
    peer_symbols = list(PEER_SYMBOLS_BY_SECTOR.get(sector, []))
    if symbol not in peer_symbols:
        peer_symbols.insert(0, symbol)
    peer_symbols = peer_symbols[:5]

    summary_rows: list[dict[str, Any]] = []
    perf_df = pd.DataFrame()

    for peer_symbol in peer_symbols:
        try:
            peer_df = fetch_stock_data(peer_symbol, start_date=start_date, end_date=end_date)
        except Exception:
            continue
        if peer_df.empty or "Close" not in peer_df.columns:
            continue

        first_close = _to_float(peer_df["Close"].iloc[0], 0.0)
        last_close = _to_float(peer_df["Close"].iloc[-1], 0.0)
        change_pct = ((last_close / first_close) - 1.0) * 100.0 if first_close > 0 else 0.0
        volatility_pct = peer_df["Close"].pct_change().std() * (252 ** 0.5) * 100.0
        avg_volume = _to_float(peer_df["Volume"].mean() if "Volume" in peer_df.columns else 0.0, 0.0)

        summary_rows.append(
            {
                "Symbol": peer_symbol,
                "Current Price": round(last_close, 2),
                "Return %": round(change_pct, 2),
                "Volatility %": round(_to_float(volatility_pct, 0.0), 2),
                "Avg Volume": round(avg_volume, 0),
            }
        )

        normalized = (peer_df["Close"] / first_close) * 100.0 if first_close > 0 else peer_df["Close"]
        perf_df[peer_symbol] = normalized.reset_index(drop=True)

    summary_df = pd.DataFrame(summary_rows)
    return summary_df, perf_df


def _sector_behavior_insight(symbol: str, peer_summary_df: pd.DataFrame) -> dict[str, Any]:
    """Summarize sector behavior and relative strength vs peers."""
    result = {
        "sector": _get_symbol_sector(symbol),
        "stock_return": 0.0,
        "sector_avg_return": 0.0,
        "relative_strength": 0.0,
        "rank_text": "N/A",
    }
    if peer_summary_df.empty or "Return %" not in peer_summary_df.columns:
        return result

    sector_avg = float(peer_summary_df["Return %"].mean())
    current_row = peer_summary_df.loc[peer_summary_df["Symbol"] == symbol]
    stock_ret = float(current_row["Return %"].iloc[0]) if not current_row.empty else 0.0
    relative = stock_ret - sector_avg

    sorted_peers = peer_summary_df.sort_values("Return %", ascending=False).reset_index(drop=True)
    rank_position = sorted_peers.index[sorted_peers["Symbol"] == symbol]
    if len(rank_position) > 0:
        rank_text = f"{int(rank_position[0]) + 1} / {len(sorted_peers)}"
    else:
        rank_text = f"- / {len(sorted_peers)}"

    result.update(
        {
            "stock_return": stock_ret,
            "sector_avg_return": sector_avg,
            "relative_strength": relative,
            "rank_text": rank_text,
        }
    )
    return result


def _render_advanced_company_insights(symbol: str, start_date: str, end_date: str) -> None:
    """Render advanced insights panel for both fast and non-fast modes."""
    st.markdown(
        "<div style='font-size:14px;color:#94a3b8;margin-bottom:12px;font-weight:600;text-transform:uppercase;letter-spacing:1px;'>Advanced Company Insights</div>",
        unsafe_allow_html=True,
    )

    company_profile = _fetch_company_profile(symbol)
    peer_summary_df, peer_perf_df = _build_peer_comparison(symbol, start_date=start_date, end_date=end_date)
    sector_stats = _sector_behavior_insight(symbol, peer_summary_df)

    insights_tab1, insights_tab2, insights_tab3, insights_tab4 = st.tabs(
        ["Company Brief", "Fundamentals", "Sector Behavior", "Peer Comparison"]
    )

    with insights_tab1:
        st.markdown(f"### {company_profile.get('name', symbol)}")
        st.caption(f"{company_profile.get('sector', 'N/A')} | {company_profile.get('industry', 'N/A')}")
        st.write(str(company_profile.get("summary") or "Company summary not available."))

    with insights_tab2:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Market Cap", _format_large_number(company_profile.get("market_cap")))
        with c2:
            st.metric("P/E", f"{_to_float(company_profile.get('pe_ratio'), 0.0):.2f}")
        with c3:
            st.metric("P/B", f"{_to_float(company_profile.get('pb_ratio'), 0.0):.2f}")
        with c4:
            dividend_yield = _to_float(company_profile.get("dividend_yield"), 0.0) * 100.0
            st.metric("Dividend Yield", f"{dividend_yield:.2f}%")

        c5, c6, c7, c8 = st.columns(4)
        with c5:
            roe = _to_float(company_profile.get("roe"), 0.0) * 100.0
            st.metric("ROE", f"{roe:.2f}%")
        with c6:
            st.metric("Debt/Equity", f"{_to_float(company_profile.get('debt_to_equity'), 0.0):.2f}")
        with c7:
            st.metric("52W High", f"₹{_to_float(company_profile.get('fifty_two_week_high'), 0.0):,.2f}")
        with c8:
            st.metric("52W Low", f"₹{_to_float(company_profile.get('fifty_two_week_low'), 0.0):,.2f}")

    with insights_tab3:
        sec_col1, sec_col2, sec_col3, sec_col4 = st.columns(4)
        with sec_col1:
            st.metric("Sector", str(sector_stats.get("sector", "Unknown")))
        with sec_col2:
            st.metric("Stock Return", f"{_to_float(sector_stats.get('stock_return'), 0.0):.2f}%")
        with sec_col3:
            st.metric("Sector Avg Return", f"{_to_float(sector_stats.get('sector_avg_return'), 0.0):.2f}%")
        with sec_col4:
            st.metric("Relative Strength", f"{_to_float(sector_stats.get('relative_strength'), 0.0):+.2f}%")

        st.caption(f"Peer return rank in sector: {sector_stats.get('rank_text', 'N/A')}")

    with insights_tab4:
        if peer_summary_df.empty:
            st.info("Peer comparison data is unavailable for this stock right now.")
        else:
            st.dataframe(peer_summary_df.sort_values("Return %", ascending=False), use_container_width=True)

            if not peer_perf_df.empty:
                line_fig = go.Figure()
                for col in peer_perf_df.columns:
                    line_fig.add_trace(
                        go.Scatter(
                            y=peer_perf_df[col],
                            mode="lines",
                            name=col,
                            line={"width": 3 if col == symbol else 1.8},
                        )
                    )
                line_fig.update_layout(
                    title="Normalized Price Comparison (Base=100)",
                    height=360,
                    margin=dict(l=10, r=10, t=40, b=10),
                    legend_title_text="Peers",
                )
                st.plotly_chart(line_fig, use_container_width=True)

    st.markdown("<div style='margin:24px 0;'></div>", unsafe_allow_html=True)


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


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_market_dashboard_snapshot() -> dict[str, Any]:
    """Fetch broad market snapshot used for the pre-analysis dashboard."""
    index_map = {
        "Sensex": "^BSESN",
        "Nifty 50": "^NSEI",
        "Nifty Bank": "^NSEBANK",
        "India VIX": "^INDIAVIX",
    }

    indices: list[dict[str, Any]] = []
    for label, ticker in index_map.items():
        try:
            history = yf.Ticker(ticker).history(period="5d", interval="1d", auto_adjust=False)
        except Exception:
            history = pd.DataFrame()

        if history.empty or "Close" not in history.columns:
            indices.append({"name": label, "value": None, "delta": None, "delta_pct": None})
            continue

        latest = _to_float(history["Close"].iloc[-1], 0.0)
        previous = _to_float(history["Close"].iloc[-2], latest) if len(history) > 1 else latest
        delta = latest - previous
        delta_pct = ((delta / previous) * 100.0) if previous else 0.0
        indices.append(
            {
                "name": label,
                "value": latest,
                "delta": delta,
                "delta_pct": delta_pct,
            }
        )

    movers: list[dict[str, Any]] = []
    for symbol, name, _sector in POPULAR_STOCKS:
        try:
            frame = yf.Ticker(symbol).history(period="3d", interval="1d", auto_adjust=False)
        except Exception:
            frame = pd.DataFrame()

        if frame.empty or "Close" not in frame.columns:
            continue

        latest_close = _to_float(frame["Close"].iloc[-1], 0.0)
        previous_close = _to_float(frame["Close"].iloc[-2], latest_close) if len(frame) > 1 else latest_close
        change = latest_close - previous_close
        change_pct = ((change / previous_close) * 100.0) if previous_close else 0.0
        latest_volume = _to_float(frame["Volume"].iloc[-1], 0.0) if "Volume" in frame.columns else 0.0

        movers.append(
            {
                "Symbol": symbol,
                "Company": name,
                "Price": round(latest_close, 2),
                "Change %": round(change_pct, 2),
                "Volume": int(latest_volume),
            }
        )

    movers_df = pd.DataFrame(movers)
    if movers_df.empty:
        return {
            "indices": indices,
            "breadth": {"advancers": 0, "decliners": 0, "unchanged": 0},
            "top_gainers": pd.DataFrame(),
            "top_losers": pd.DataFrame(),
            "most_active": pd.DataFrame(),
        }

    advancers = int((movers_df["Change %"] > 0).sum())
    decliners = int((movers_df["Change %"] < 0).sum())
    unchanged = int((movers_df["Change %"] == 0).sum())
    gainers = movers_df.sort_values("Change %", ascending=False).head(5).reset_index(drop=True)
    losers = movers_df.sort_values("Change %", ascending=True).head(5).reset_index(drop=True)
    active = movers_df.sort_values("Volume", ascending=False).head(5).reset_index(drop=True)

    return {
        "indices": indices,
        "breadth": {"advancers": advancers, "decliners": decliners, "unchanged": unchanged},
        "top_gainers": gainers,
        "top_losers": losers,
        "most_active": active,
    }


def _render_pre_analysis_market_dashboard() -> None:
    """Render market dashboard widgets while waiting for user analysis input."""
    snapshot = _fetch_market_dashboard_snapshot()

    st.markdown(
        "<div style='font-size:14px;color:#94a3b8;margin:8px 0 12px 0;font-weight:600;text-transform:uppercase;letter-spacing:1px;'>Live Market Snapshot</div>",
        unsafe_allow_html=True,
    )

    index_cols = st.columns(4)
    for col, item in zip(index_cols, snapshot.get("indices", [])):
        with col:
            value = item.get("value")
            delta = item.get("delta")
            delta_pct = item.get("delta_pct")
            if value is None or delta is None or delta_pct is None:
                st.metric(str(item.get("name", "Index")), "N/A", "Data unavailable")
            else:
                st.metric(
                    str(item.get("name", "Index")),
                    f"{float(value):,.2f}",
                    f"{float(delta):+,.2f} ({float(delta_pct):+.2f}%)",
                )

    st.markdown("<div style='margin:10px 0;'></div>", unsafe_allow_html=True)

    breadth = snapshot.get("breadth", {})
    breadth_col1, breadth_col2, breadth_col3 = st.columns(3)
    breadth_col1.metric("Advancers", f"{int(breadth.get('advancers', 0))}")
    breadth_col2.metric("Decliners", f"{int(breadth.get('decliners', 0))}")
    breadth_col3.metric("Unchanged", f"{int(breadth.get('unchanged', 0))}")

    st.markdown("<div style='margin:10px 0;'></div>", unsafe_allow_html=True)

    table_col1, table_col2, table_col3 = st.columns(3)
    with table_col1:
        st.markdown("#### Top Gainers")
        gainers_df = snapshot.get("top_gainers", pd.DataFrame())
        if isinstance(gainers_df, pd.DataFrame) and not gainers_df.empty:
            st.dataframe(gainers_df[["Symbol", "Price", "Change %"]], use_container_width=True, hide_index=True)
        else:
            st.info("Gainers data unavailable.")
    with table_col2:
        st.markdown("#### Top Losers")
        losers_df = snapshot.get("top_losers", pd.DataFrame())
        if isinstance(losers_df, pd.DataFrame) and not losers_df.empty:
            st.dataframe(losers_df[["Symbol", "Price", "Change %"]], use_container_width=True, hide_index=True)
        else:
            st.info("Losers data unavailable.")
    with table_col3:
        st.markdown("#### Most Active")
        active_df = snapshot.get("most_active", pd.DataFrame())
        if isinstance(active_df, pd.DataFrame) and not active_df.empty:
            st.dataframe(active_df[["Symbol", "Price", "Volume"]], use_container_width=True, hide_index=True)
        else:
            st.info("Most active data unavailable.")


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_indian_symbol_suggestions(query: str, limit: int = 20) -> list[dict[str, str]]:
    """Fetch Indian equity symbol suggestions from Yahoo Finance search API."""
    text = str(query or "").strip()
    if not text:
        return []

    url = (
        "https://query2.finance.yahoo.com/v1/finance/search"
        f"?q={quote_plus(text)}&quotesCount=50&newsCount=0"
    )
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})

    try:
        with urlopen(request, timeout=4) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (URLError, TimeoutError, ValueError, json.JSONDecodeError):
        return []

    quotes = payload.get("quotes", []) if isinstance(payload, dict) else []
    if not isinstance(quotes, list):
        return []

    entries: list[dict[str, str]] = []
    seen_symbols: set[str] = set()
    query_upper = text.upper()

    for item in quotes:
        if not isinstance(item, dict):
            continue

        symbol = str(item.get("symbol", "")).upper().strip()
        if not symbol or symbol in seen_symbols:
            continue

        # Keep Indian exchange symbols only.
        if not (symbol.endswith(".NS") or symbol.endswith(".BO")):
            continue

        quote_type = str(item.get("quoteType", "")).upper().strip()
        if quote_type and quote_type != "EQUITY":
            continue

        name = str(item.get("shortname") or item.get("longname") or symbol).strip()
        exchange = str(item.get("exchangeDisp") or item.get("exchange") or "").strip()
        label = f"{symbol} | {name}" if not exchange else f"{symbol} | {name} ({exchange})"

        entries.append({"symbol": symbol, "label": label})
        seen_symbols.add(symbol)

    # Prioritize symbols and names starting with current query text.
    entries.sort(
        key=lambda row: (
            not row["symbol"].startswith(query_upper),
            not row["label"].upper().startswith(query_upper),
            row["symbol"],
        )
    )
    return entries[: max(1, int(limit))]


@st.cache_resource(show_spinner=False)
def _initialize_sector_models() -> dict[str, Any]:
    """Automatically train sector-aware models on app startup for maximum accuracy."""
    manifest_path = "models/sector/manifest.json"
    router_bundle_path = "models/sector/router_bundle.pkl"
    
    # Check if models already exist
    if os.path.exists(manifest_path) and os.path.exists(router_bundle_path):
        try:
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
            return {"status": "loaded", "manifest": manifest, "message": "Using cached models"}
        except Exception:
            pass
    
    # Train new models
    try:
        with st.spinner("🚀 Initializing AI models for market analysis (one-time setup)..."):
            train_result = train_sector_models(
                symbols=DEFAULT_NSE_STOCKS,
                period="1y",
                model_type="xgboost",
                min_rows_per_sector=100,
                min_symbols_per_sector=1,
                save_dir="models/sector",
                run_backtests=False,
            )
            
            metrics = train_result.get("metrics", {}) if isinstance(train_result, dict) else {}
            return {
                "status": "trained",
                "manifest": metrics,
                "message": f"Models trained: {int(metrics.get('num_trained_sectors', 0))} sectors ready",
            }
    except Exception as exc:
        st.warning(f"Model initialization: {str(exc)[:100]}")
        return {"status": "error", "message": str(exc)}


logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
page_icon = logo_path if os.path.exists(logo_path) else "📈"

header_logo_html = "<span class='header-logo-fallback'>📈</span>"
if os.path.exists(logo_path):
    with open(logo_path, "rb") as logo_file:
        logo_b64 = base64.b64encode(logo_file.read()).decode("ascii")
    header_logo_html = (
        f"<img class='header-logo' src='data:image/png;base64,{logo_b64}' alt='Stock Market AI Logo' />"
    )

st.set_page_config(page_title="Stock Market AI", page_icon=page_icon, layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
        :root {
            --bg-1: #0b1220;
            --bg-2: #111b2e;
            --panel: rgba(17, 27, 46, 0.74);
            --panel-border: rgba(148, 163, 184, 0.22);
            --text-main: #e2e8f0;
            --text-muted: #94a3b8;
            --accent: #0ea5a4;
            --accent-2: #06b6d4;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: #000000 !important;
        }
        .stApp {
            background: #000000 !important;
        }
        .header-container {
            background: linear-gradient(140deg, rgba(15,23,42,0.9) 0%, rgba(15,23,42,0.5) 100%);
            padding: 26px 28px;
            border-bottom: 1px solid var(--panel-border);
            margin: -80px -40px 0 -40px;
            backdrop-filter: blur(10px);
        }
        .header-brand {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .header-logo {
            height: 34px;
            width: 34px;
            object-fit: contain;
            border-radius: 6px;
        }
        .header-logo-fallback {
            font-size: 30px;
            line-height: 1;
        }
        .header-title {
            font-size: 34px;
            font-weight: 900;
            color: var(--text-main);
            margin: 0;
            background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            letter-spacing: -0.6px;
        }
        .header-subtitle {
            font-size: 14px;
            color: var(--text-muted);
            margin-top: 6px;
        }
        [data-testid="block-container"] {
            max-width: 1500px;
            padding-top: 1.2rem;
            padding-bottom: 2.2rem;
        }
        .stMarkdown p {
            color: var(--text-main);
        }
        .stAlert {
            border-radius: 12px !important;
            border: 1px solid var(--panel-border) !important;
            background: rgba(15, 23, 42, 0.65) !important;
        }
        div[data-testid="stExpander"] {
            border: 1px solid var(--panel-border) !important;
            border-radius: 14px !important;
            background: rgba(15, 23, 42, 0.55) !important;
            overflow: hidden;
        }
        div[data-testid="stExpander"] details summary {
            background: rgba(15, 23, 42, 0.45) !important;
            padding: 8px 10px;
        }
        .metric-card {
            background: linear-gradient(165deg, rgba(30,41,59,0.75), rgba(15,23,42,0.88));
            border: 1px solid var(--panel-border);
            border-radius: 14px;
            padding: 14px;
            box-shadow: 0 8px 24px rgba(2, 6, 23, 0.25);
        }
        .command-center {
            background: linear-gradient(130deg, rgba(14,165,164,0.15), rgba(6,182,212,0.10));
            border: 1px solid rgba(6,182,212,0.35);
            border-radius: 14px;
            padding: 14px;
            margin-bottom: 16px;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.25);
        }
        .command-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px;
            margin-top: 10px;
        }
        .command-cell {
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148,163,184,0.25);
            border-radius: 10px;
            padding: 10px 12px;
        }
        .command-label {
            font-size: 11px;
            text-transform: uppercase;
            color: #94a3b8;
            letter-spacing: 0.6px;
            margin-bottom: 5px;
        }
        .command-value {
            font-size: 18px;
            font-weight: 700;
            color: #f8fafc;
        }
        .card {
            background: var(--panel);
            border: 1px solid var(--panel-border);
            border-radius: 14px;
            padding: 20px;
            backdrop-filter: blur(12px);
            margin-bottom: 16px;
        }
        .card-title {
            font-size: 12px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }
        .card-value {
            font-size: 28px;
            font-weight: 700;
            color: var(--text-main);
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
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        .stNumberInput div[data-baseweb="input"] > div {
            border-radius: 12px !important;
            border: 1px solid var(--panel-border) !important;
            background: rgba(15, 23, 42, 0.7) !important;
        }
        .stButton > button {
            border-radius: 12px !important;
            border: 1px solid rgba(6,182,212,0.35) !important;
            background: linear-gradient(135deg, rgba(6,182,212,0.25), rgba(14,165,164,0.25)) !important;
            color: #e6fffb !important;
            font-weight: 700 !important;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(6,182,212,0.2);
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 10px 10px 0 0;
            background: rgba(15,23,42,0.55);
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(14,165,164,0.25), rgba(6,182,212,0.25)) !important;
            color: #d1fae5 !important;
            border-bottom: 2px solid rgba(6,182,212,0.8) !important;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--panel-border);
            border-radius: 12px;
            overflow: hidden;
            background: rgba(15, 23, 42, 0.55);
        }
        div[data-testid="stMetric"] {
            border: 1px solid rgba(148,163,184,0.22);
            border-radius: 12px;
            padding: 8px;
            background: rgba(15, 23, 42, 0.55);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="header-container">
        <div class="header-brand">
            {header_logo_html}
            <h1 class="header-title">Stock Market AI</h1>
        </div>
        <p class="header-subtitle">Professional trading insights powered by machine learning</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Initialize sector models automatically on startup
_initialize_sector_models()

# Top control bar
col1, col2, col3, col4, col5, col6 = st.columns([2, 1.2, 1.2, 1.2, 1.0, 1])
with col1:
    # Create professional stock selector dropdown with popular stocks
    stock_options = []
    sector_groups = {}
    for symbol, name, sector in POPULAR_STOCKS:
        if sector not in sector_groups:
            sector_groups[sector] = []
        display_text = f"{'🏢' if sector == 'Banking' else '💻' if sector == 'IT' else '⚡' if sector == 'Energy' else '🚗' if sector == 'Automobiles' else '💊' if sector == 'Pharma' else '🏗️'} {symbol} - {name}"
        sector_groups[sector].append((display_text, symbol))
    
    # Flatten with sector headers
    for sector in sorted(sector_groups.keys()):
        stock_options.append(f"─ {sector}")
        for display_text, _ in sector_groups[sector]:
            stock_options.append(display_text)
    
    default_index = 0
    default_symbol = st.session_state.get("selected_stock", "RELIANCE.NS")
    for idx, option in enumerate(stock_options):
        if default_symbol in option:
            default_index = idx
            break
    
    selected_option = st.selectbox(
        "Popular Stocks",
        options=stock_options,
        index=default_index,
        label_visibility="collapsed",
        format_func=lambda x: x,
    )
    
    # Extract symbol from selected option (skip sector headers)
    selected_option_str = str(selected_option or "").strip()
    if selected_option_str.startswith("─"):
        symbol = default_symbol
    else:
        # Extract symbol from display text (format: "emoji SYMBOL - Name")
        parts = selected_option_str.split(" - ")
        if len(parts) > 0:
            symbol_part = parts[0].strip()
            # Remove emoji prefix
            symbol = symbol_part.split()[-1] if symbol_part else default_symbol
        else:
            symbol = default_symbol
    
    st.session_state["selected_stock"] = symbol
    
    # Alternative: Search for other stocks via autocomplete
    with st.expander("🔍 Search Other Stocks"):
        ticker_query_input = st.text_input(
            "Search by ticker or company name",
            placeholder="e.g., 't' shows TCS, TATAMOTORS | 'reliance' for Reliance Industries",
            label_visibility="collapsed",
            key="ticker_search_input",
        )
        ticker_query = str(ticker_query_input or "").strip()
        
        if ticker_query and len(ticker_query) >= 1:
            suggestions = _fetch_indian_symbol_suggestions(ticker_query)
            if suggestions:
                suggestion_labels = [f"🔎 {item['label']}" for item in suggestions]
                selected_label = st.selectbox(
                    "Search Results",
                    options=suggestion_labels,
                    label_visibility="collapsed",
                    key="ticker_suggestion_select",
                )
                symbol_map = {f"🔎 {item['label']}": item["symbol"] for item in suggestions}
                symbol = symbol_map.get(selected_label, symbol)
                st.session_state["selected_stock"] = symbol
                st.caption(f"✓ Selected: **{symbol}**")
            else:
                st.caption("❌ No results found. Try different search terms.")
with col2:
    market_open_now = _is_nse_market_open_now()
    available_timeframes = ["Intraday", "1 Day", "1 Week", "1 Month", "6 Months", "1 Year", "2 Years"]
    if not market_open_now:
        available_timeframes = [tf for tf in available_timeframes if tf != "Intraday"]
    default_tf_index = available_timeframes.index("1 Day") if "1 Day" in available_timeframes else 0
    timeframe = st.selectbox("Timeframe", available_timeframes, index=default_tf_index, label_visibility="collapsed")
with col3:
    investment = st.number_input("Investment (₹)", min_value=1000.0, value=100000.0, step=10000.0, label_visibility="collapsed")
with col4:
    strategy = st.selectbox("Strategy", ["Conservative", "Moderate", "Aggressive"], index=1, label_visibility="collapsed")
with col5:
    fast_mode = st.toggle("Fast Mode", value=True, help="Skips slower components like backtesting.")
with col6:
    analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

analysis_params = {
    "symbol": symbol,
    "timeframe": timeframe,
    "investment": float(investment),
    "strategy": strategy,
    "fast_mode": bool(fast_mode),
}

if analyze_btn:
    st.session_state["analysis_active"] = True
    st.session_state["analysis_params"] = dict(analysis_params)

if not st.session_state.get("analysis_active", False):
    st.markdown(
        """
        <div style="text-align: center; padding: 60px 20px; color: #94a3b8;">
            <p style="font-size: 18px; margin-bottom: 10px;">👆 Enter stock ticker and click Analyze to get started</p>
            <p style="font-size: 14px;">Real-time AI predictions • Multi-timeframe analysis • Backtest performance</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _render_pre_analysis_market_dashboard()
    st.stop()

if not symbol:
    st.error("Please provide a stock ticker symbol.")
    st.stop()

cached_payload = st.session_state.get("analysis_payload")
should_run_analysis = analyze_btn or not isinstance(cached_payload, dict)

if should_run_analysis:
    today = pd.Timestamp.today().normalize()
    lookback_days = TIMEFRAME_MAP[timeframe]
    start_date = (today - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")

    spinner_text = "Analyzing stock with Fast Mode..." if fast_mode else "Analyzing stock and running backtests..."
    with st.spinner(spinner_text):
        try:
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
            st.exception(exc)
            st.stop()

    st.session_state["analysis_payload"] = {
        "summary": summary,
        "run": run,
        "chart_df": chart_df,
        "start_date": start_date,
        "end_date": end_date,
    }
else:
    payload = st.session_state.get("analysis_payload")
    if not isinstance(payload, dict):
        st.session_state["analysis_active"] = False
        st.rerun()

    summary = payload["summary"]
    run = payload["run"]
    chart_df = payload["chart_df"]
    start_date = str(payload["start_date"])
    end_date = str(payload["end_date"])

    cached_params = st.session_state.get("analysis_params", {})
    if isinstance(cached_params, dict):
        symbol = str(cached_params.get("symbol", symbol))
        timeframe = str(cached_params.get("timeframe", timeframe))
        investment = float(cached_params.get("investment", investment))
        strategy = str(cached_params.get("strategy", strategy))
        fast_mode = bool(cached_params.get("fast_mode", fast_mode))

st.success("Analysis completed successfully.")

dataset = run.get("dataset", pd.DataFrame())

default_confidence = float(summary["confidence"])
confidence = _to_float(run.get("latest_probability_increase"), default_confidence)
confidence = max(0.0, min(1.0, confidence))

timeframe_prediction = "UP" if confidence >= 0.5 else "DOWN"
decision = _prediction_to_signal(timeframe_prediction)

levels = dict(summary["trade_levels"])
if not dataset.empty and {"Close", "ATR_14"}.issubset(dataset.columns):
    try:
        levels = dict(
            generate_trade_levels(
                df=dataset,
                prediction=timeframe_prediction,
                probability=confidence,
            )
        )
    except Exception:
        levels = dict(summary["trade_levels"])

entry_price = _to_float(levels.get("entry"), 0.0)
target_price = _to_float(levels.get("target"), 0.0)
stop_loss_price = _to_float(levels.get("stop_loss"), 0.0)

capital_value = float(investment)
position_size = (capital_value / entry_price) if entry_price > 0 else 0.0
max_profit_per_share = abs(target_price - entry_price)
max_loss_per_share = abs(entry_price - stop_loss_price)
max_profit_rs = float(max_profit_per_share * position_size)
max_loss_rs = float(max_loss_per_share * position_size)
profit_percent = float((max_profit_rs / capital_value) * 100.0) if capital_value > 0 else 0.0
loss_percent = float((max_loss_rs / capital_value) * 100.0) if capital_value > 0 else 0.0
pnl = {
    "max_profit_rs": max_profit_rs,
    "max_loss_rs": max_loss_rs,
    "profit_percent": profit_percent,
    "loss_percent": loss_percent,
}

risk_level = str(summary["risk_level"])
if not dataset.empty and {"Close", "ATR_14"}.issubset(dataset.columns):
    try:
        latest = dataset.tail(1)
        atr_val = _to_float(latest["ATR_14"].iloc[0], 0.0)
        close_val = _to_float(latest["Close"].iloc[0], 0.0)
        volatility = (atr_val / close_val) if close_val > 0 else 0.0
        risk_level = classify_risk(probability=confidence, volatility=volatility)
    except Exception:
        risk_level = str(summary["risk_level"])

# Top-most insights section (visible in both Fast and Non-Fast modes)
_render_advanced_company_insights(symbol=symbol, start_date=start_date, end_date=end_date)

if fast_mode and not dataset.empty and callable(predict_with_saved_sector_router):
    try:
        latest_model_features = run.get("feature_columns", [])
        if latest_model_features:
            latest_features = dataset[list(latest_model_features)].tail(1)
            router_result = predict_with_saved_sector_router(
                symbol=symbol,
                latest_features=latest_features,
                save_dir="models/sector",
                blend_mode="fixed",
                fixed_sector_weight=0.65,
                min_sector_confidence=0.55,
            )
            if router_result is not None:
                confidence = float(router_result["probability_up"])
                decision = "BUY" if int(router_result["prediction"]) == 1 else "SELL"
    except Exception:
        pass

mode_label = "Fast" if fast_mode else "Deep"
signal_color = SIGNAL_COLORS.get(decision, "#f59e0b")
st.markdown(
        f"""
        <div class="command-center">
            <div style="display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap;">
                <div>
                    <div style="font-size:13px;color:#94a3b8;text-transform:uppercase;letter-spacing:0.7px;">Trading Command Center</div>
                    <div style="font-size:22px;font-weight:800;color:#f8fafc;">{symbol}</div>
                </div>
                <div style="display:flex;gap:8px;align-items:center;">
                    <span style="padding:4px 10px;border-radius:999px;border:1px solid rgba(148,163,184,0.35);color:#cbd5e1;font-size:12px;">Mode: {mode_label}</span>
                    <span style="padding:4px 10px;border-radius:999px;background:{signal_color};color:#021014;font-weight:700;font-size:12px;">Signal: {decision}</span>
                </div>
            </div>
            <div class="command-grid">
                <div class="command-cell">
                    <div class="command-label">Confidence</div>
                    <div class="command-value">{confidence:.1%}</div>
                    <div style="margin-top:6px;height:7px;border-radius:999px;background:#0f172a;overflow:hidden;border:1px solid rgba(148,163,184,0.2);">
                        <div style="height:100%;width:{int(max(0, min(100, confidence * 100)))}%;background:{signal_color};"></div>
                    </div>
                </div>
                <div class="command-cell">
                    <div class="command-label">Risk Profile</div>
                    <div class="command-value">{risk_level}</div>
                </div>
                <div class="command-cell">
                    <div class="command-label">Capital</div>
                    <div class="command-value">₹{float(investment):,.0f}</div>
                </div>
                <div class="command-cell">
                    <div class="command-label">Horizon</div>
                    <div class="command-value">{timeframe}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
)

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
    "1 Day": summary["timeframes"].get("1d", "N/A"),
    "1 Week": "N/A",
    "1 Month": summary["timeframes"].get("1m", "N/A"),
    "6 Months": summary["timeframes"].get("6m", "N/A"),
    "1 Year": summary["timeframes"].get("1y", "N/A"),
}
if _is_nse_market_open_now():
    timeframes = {"Intraday": summary["timeframes"].get("intraday", "N/A"), **timeframes}

tf_cols = st.columns(len(timeframes))
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
    control_col1, control_col2, control_col3, control_col4, control_col5, control_col6 = st.columns([1.2, 1.0, 1.0, 1.0, 1.2, 1.0])
    with control_col1:
        chart_mode = st.selectbox(
            "Chart Type",
            ["Candlestick", "Line"],
            index=0,
            label_visibility="collapsed",
            key="price_chart_mode",
        )
    with control_col2:
        show_volume = st.toggle("Volume", value=True, key="price_chart_show_volume")
    with control_col3:
        show_mas = st.toggle("MA 20/50", value=True, key="price_chart_show_ma")
    with control_col4:
        show_levels = st.toggle("Trade Levels", value=True, key="price_chart_show_levels")
    with control_col5:
        chart_range = st.selectbox(
            "Range",
            ["All", "1W", "1M", "3M", "YTD", "1Y"],
            index=0,
            label_visibility="collapsed",
            key="price_chart_range",
        )
    with control_col6:
        show_atr_bands = st.toggle("ATR Bands", value=False, key="price_chart_show_atr_bands")

    plot_df = chart_df.copy()
    plot_df["Date"] = pd.to_datetime(plot_df["Date"], errors="coerce")
    plot_df = plot_df.sort_values("Date").reset_index(drop=True)

    if chart_range != "All" and not plot_df.empty:
        max_date = pd.to_datetime(plot_df["Date"]).max()
        if chart_range == "1W":
            min_date = max_date - pd.Timedelta(days=7)
            plot_df = plot_df[plot_df["Date"] >= min_date]
        elif chart_range == "1M":
            min_date = max_date - pd.Timedelta(days=30)
            plot_df = plot_df[plot_df["Date"] >= min_date]
        elif chart_range == "3M":
            min_date = max_date - pd.Timedelta(days=90)
            plot_df = plot_df[plot_df["Date"] >= min_date]
        elif chart_range == "YTD":
            min_date = pd.Timestamp(year=max_date.year, month=1, day=1)
            plot_df = plot_df[plot_df["Date"] >= min_date]
        elif chart_range == "1Y":
            min_date = max_date - pd.Timedelta(days=365)
            plot_df = plot_df[plot_df["Date"] >= min_date]

    plot_df = plot_df.reset_index(drop=True)
    plot_df["MA_20"] = plot_df["Close"].rolling(window=20, min_periods=1).mean()
    plot_df["MA_50"] = plot_df["Close"].rolling(window=50, min_periods=1).mean()
    true_range = (plot_df["High"] - plot_df["Low"]).abs()
    plot_df["ATR_14"] = true_range.rolling(window=14, min_periods=1).mean()
    plot_df["ATR_UPPER"] = plot_df["Close"] + plot_df["ATR_14"]
    plot_df["ATR_LOWER"] = plot_df["Close"] - plot_df["ATR_14"]

    if show_volume:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.72, 0.28],
        )
    else:
        fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

    if chart_mode == "Candlestick":
        fig.add_trace(
            go.Candlestick(
                x=plot_df["Date"],
                open=plot_df["Open"],
                high=plot_df["High"],
                low=plot_df["Low"],
                close=plot_df["Close"],
                name=symbol,
                increasing_line_color="#22c55e",
                decreasing_line_color="#ef4444",
                increasing_fillcolor="#14532d",
                decreasing_fillcolor="#7f1d1d",
                whiskerwidth=0.7,
            ),
            row=1,
            col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=plot_df["Date"],
                y=plot_df["Close"],
                mode="lines",
                name=f"{symbol} Close",
                line=dict(color="#38bdf8", width=2.5),
            ),
            row=1,
            col=1,
        )

    if show_mas:
        fig.add_trace(
            go.Scatter(
                x=plot_df["Date"],
                y=plot_df["MA_20"],
                mode="lines",
                name="MA 20",
                line=dict(color="#f59e0b", width=1.6),
            ),
            row=1,
            col=1,
        )

    if show_atr_bands:
        fig.add_trace(
            go.Scatter(
                x=plot_df["Date"],
                y=plot_df["ATR_UPPER"],
                mode="lines",
                name="ATR Upper",
                line=dict(color="rgba(56,189,248,0.55)", width=1.2),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=plot_df["Date"],
                y=plot_df["ATR_LOWER"],
                mode="lines",
                name="ATR Lower",
                line=dict(color="rgba(56,189,248,0.55)", width=1.2),
                fill="tonexty",
                fillcolor="rgba(56,189,248,0.10)",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=plot_df["Date"],
                y=plot_df["MA_50"],
                mode="lines",
                name="MA 50",
                line=dict(color="#a78bfa", width=1.6),
            ),
            row=1,
            col=1,
        )

    if show_volume and "Volume" in plot_df.columns:
        volume_colors = [
            "rgba(34,197,94,0.55)" if close_val >= open_val else "rgba(239,68,68,0.55)"
            for open_val, close_val in zip(plot_df["Open"], plot_df["Close"])
        ]
        fig.add_trace(
            go.Bar(
                x=plot_df["Date"],
                y=plot_df["Volume"],
                name="Volume",
                marker_color=volume_colors,
            ),
            row=2,
            col=1,
        )

    if show_levels:
        fig.add_hline(y=entry_price, line_color="#2563eb", line_width=1.4, annotation_text="Entry")
        fig.add_hline(y=target_price, line_color="#16a34a", line_width=1.4, annotation_text="Target")
        fig.add_hline(y=stop_loss_price, line_color="#dc2626", line_width=1.4, annotation_text="Stop")

    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.15)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.12)", zeroline=False)
    fig.update_layout(
        height=560 if show_volume else 470,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.35)",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

st.markdown("<div style='margin:24px 0;'></div>", unsafe_allow_html=True)
if not fast_mode:
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

# Footer Section (visible on all pages)
st.markdown("<div style='margin:60px 0 30px 0;border-top:1px solid rgba(148,163,184,0.2);padding-top:30px;'></div>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align:center;color:#94a3b8;padding:20px;border-top:1px solid rgba(148,163,184,0.15);">
        <div style="margin-bottom:16px;">
            <p style="font-size:12px;color:#cbd5e1;margin-bottom:12px;">
                <strong>Stock Market AI</strong> • Professional Trading Insights Powered by Machine Learning
            </p>
        </div>
        <div style="display:flex;justify-content:center;gap:20px;flex-wrap:wrap;margin-bottom:16px;">
            <a href="mailto:bhatianirudhsingh592@gmail.com" style="color:#06b6d4;text-decoration:none;font-size:13px;font-weight:500;">
                📧 Email: bhatianirudhsingh592@gmail.com
            </a>
            <a href="https://github.com/anirudhsinghbhati" target="_blank" style="color:#06b6d4;text-decoration:none;font-size:13px;font-weight:500;">
                GitHub: github.com/anirudhsinghbhati
            </a>
            <a href="https://www.linkedin.com/in/anirudh-singh-bhati-0a4455274/" target="_blank" style="color:#06b6d4;text-decoration:none;font-size:13px;font-weight:500;">
                LinkedIn: linkedin.com/in/anirudhsinghbhati
            </a>
        </div>
        <div style="border-top:1px solid rgba(148,163,184,0.15);padding-top:12px;margin-top:12px;">
            <p style="font-size:11px;color:#64748b;margin:4px 0;">
                Built with Streamlit | XGBoost | Scikit-learn | Technical Analysis
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
