from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict

import pandas as pd
import pytz
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

try:
    from src.train import train_stock_model
except ImportError:
    from train import train_stock_model


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def _default_date_range(days_back: int = 365 * 5) -> tuple[str, str]:
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days_back)
    return start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")


def run_daily_prediction(
    symbol: str,
    model_family: str,
    buy_threshold: float,
    sell_threshold: float,
    sl_stop: float,
    tp_stop: float,
    output_dir: str = "data/realtime",
) -> Dict[str, Any]:
    """Train on latest data and persist daily prediction payload."""
    start_date, end_date = _default_date_range(days_back=365 * 5)
    logger.info("Running daily prediction for %s (%s to %s)", symbol, start_date, end_date)

    result = train_stock_model(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        sl_stop=sl_stop,
        tp_stop=tp_stop,
        model_family=model_family,
        save_models=True,
    )

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    payload: Dict[str, Any] = {
        "timestamp": timestamp,
        "symbol": symbol,
        "model_family": model_family,
        "probability_increase": float(result.get("latest_probability_increase", 0.5)),
        "expected_return": float(result.get("latest_expected_return", 0.0)),
        "regime": str(result.get("latest_multiclass_label", "Neutral")),
        "trade_signal": str(result.get("trade_signal", "HOLD")),
        "stop_loss": result.get("stop_loss"),
        "take_profit": result.get("take_profit"),
        "metrics": result.get("metrics", {}),
        "backtest_vectorbt": result.get("backtest_vectorbt", {}),
        "backtest_backtrader": result.get("backtest_backtrader", {}),
    }

    json_path = os.path.join(output_dir, f"prediction_{symbol}_{timestamp}.json")
    latest_json_path = os.path.join(output_dir, f"latest_{symbol}.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    with open(latest_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    csv_path = os.path.join(output_dir, f"predictions_{symbol}.csv")
    row = pd.DataFrame([payload])
    if os.path.exists(csv_path):
        row.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        row.to_csv(csv_path, index=False)

    logger.info("Saved prediction artifacts: %s, %s, %s", json_path, latest_json_path, csv_path)
    return payload


def start_scheduler(
    symbol: str = "RELIANCE.NS",
    model_family: str = "ensemble",
    buy_threshold: float = 0.7,
    sell_threshold: float = 0.3,
    sl_stop: float = 0.02,
    tp_stop: float = 0.04,
    timezone_name: str = "Asia/Kolkata",
    hour: int = 18,
    minute: int = 0,
) -> None:
    """Start cron-like daily scheduler for automated prediction jobs."""
    tz = pytz.timezone(timezone_name)
    scheduler = BlockingScheduler(timezone=tz)

    trigger = CronTrigger(hour=hour, minute=minute)
    scheduler.add_job(
        run_daily_prediction,
        trigger=trigger,
        kwargs={
            "symbol": symbol,
            "model_family": model_family,
            "buy_threshold": buy_threshold,
            "sell_threshold": sell_threshold,
            "sl_stop": sl_stop,
            "tp_stop": tp_stop,
        },
        id=f"daily_prediction_{symbol}",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )

    logger.info(
        "Scheduler started for %s at %02d:%02d (%s)",
        symbol,
        hour,
        minute,
        timezone_name,
    )

    # Optional immediate run on startup.
    run_daily_prediction(
        symbol=symbol,
        model_family=model_family,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        sl_stop=sl_stop,
        tp_stop=tp_stop,
    )

    scheduler.start()


if __name__ == "__main__":
    # Example: run daily at 18:00 Asia/Kolkata.
    start_scheduler(
        symbol=os.getenv("SYMBOL", "RELIANCE.NS"),
        model_family=os.getenv("MODEL_FAMILY", "ensemble"),
        buy_threshold=float(os.getenv("BUY_THRESHOLD", "0.7")),
        sell_threshold=float(os.getenv("SELL_THRESHOLD", "0.3")),
        sl_stop=float(os.getenv("SL_STOP", "0.02")),
        tp_stop=float(os.getenv("TP_STOP", "0.04")),
        timezone_name=os.getenv("TIMEZONE", "Asia/Kolkata"),
        hour=int(os.getenv("RUN_HOUR", "18")),
        minute=int(os.getenv("RUN_MINUTE", "0")),
    )
