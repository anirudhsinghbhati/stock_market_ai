from __future__ import annotations

import os

from realtime_scheduler import run_daily_prediction


if __name__ == "__main__":
    payload = run_daily_prediction(
        symbol=os.getenv("SYMBOL", "RELIANCE.NS"),
        model_family=os.getenv("MODEL_FAMILY", "ensemble"),
        buy_threshold=float(os.getenv("BUY_THRESHOLD", "0.7")),
        sell_threshold=float(os.getenv("SELL_THRESHOLD", "0.3")),
        sl_stop=float(os.getenv("SL_STOP", "0.02")),
        tp_stop=float(os.getenv("TP_STOP", "0.04")),
    )
    print(payload)
