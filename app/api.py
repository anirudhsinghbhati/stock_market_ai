from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
import joblib
from pydantic import BaseModel, Field

# Allow app execution with: uvicorn app.api:app --reload
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import predict_probability
from src.train import (
    MULTICLASS_LABELS,
    build_training_dataset,
    predict_latest_direction,
    prepare_training_matrices,
    train_stock_model,
)


app = FastAPI(
    title="Stock AI REST API",
    version="1.0.0",
    description="REST API for stock model training, prediction, and metrics.",
)

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
RETURN_MODEL_PATH = os.path.join(MODEL_DIR, "return_model.pkl")
MULTICLASS_MODEL_PATH = os.path.join(MODEL_DIR, "multiclass_model.pkl")
API_STATE_PATH = os.path.join(MODEL_DIR, "api_state.json")


class TrainRequest(BaseModel):
    symbol: str = Field(default="RELIANCE.NS", min_length=1)
    start_date: str = Field(default="2020-01-01", description="YYYY-MM-DD")
    end_date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"), description="YYYY-MM-DD")
    model_family: str = Field(default="ensemble", description="ensemble | informer | tft")
    buy_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    sell_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    sl_stop: float = Field(default=0.02, gt=0.0, le=1.0)
    tp_stop: float = Field(default=0.04, gt=0.0, le=2.0)
    save_models: bool = Field(default=True)


class PredictRequest(BaseModel):
    retrain_if_missing: bool = Field(default=False)
    train_config: Optional[TrainRequest] = None


class APIState:
    result: Optional[Any] = None
    trained_at: Optional[str] = None
    train_config: Optional[TrainRequest] = None
    metrics: Optional[Dict[str, float]] = None


STATE = APIState()


def _default_start_date(years_back: int = 5) -> str:
    return (datetime.now() - timedelta(days=365 * years_back)).strftime("%Y-%m-%d")


def _compute_trade_levels(
    prob_up: float,
    current_price: float,
    atr_14: float | None,
    buy_threshold: float,
    sell_threshold: float,
    sl_stop: float,
    tp_stop: float,
) -> tuple[str, float | None, float | None]:
    if current_price <= 0:
        return "HOLD", None, None

    atr_value = float(atr_14) if atr_14 is not None and atr_14 > 0 else current_price * 0.01
    risk_unit = max(atr_value, current_price * max(sl_stop, 0.001))
    rr = tp_stop / max(sl_stop, 1e-6)

    if prob_up > buy_threshold:
        return "BUY", float(current_price - risk_unit), float(current_price + rr * risk_unit)

    if prob_up < sell_threshold:
        return "SELL", float(current_price + risk_unit), float(current_price - rr * risk_unit)

    return "HOLD", None, None


def _save_api_state() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    payload = {
        "trained_at": STATE.trained_at,
        "train_config": STATE.train_config.model_dump() if STATE.train_config else None,
        "metrics": STATE.metrics or {},
    }
    with open(API_STATE_PATH, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def _load_api_state() -> Dict[str, Any]:
    if not os.path.exists(API_STATE_PATH):
        return {}
    with open(API_STATE_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def _load_persisted_models() -> tuple[Any, Any, Any]:
    missing = [
        path
        for path in [MODEL_PATH, RETURN_MODEL_PATH, MULTICLASS_MODEL_PATH]
        if not os.path.exists(path)
    ]
    if missing:
        missing_names = ", ".join(os.path.basename(path) for path in missing)
        raise HTTPException(status_code=400, detail=f"Persisted model files not found: {missing_names}")

    model = joblib.load(MODEL_PATH)
    return_model = joblib.load(RETURN_MODEL_PATH)
    multiclass_model = joblib.load(MULTICLASS_MODEL_PATH)
    return model, return_model, multiclass_model


def _resolve_base_config(payload: PredictRequest) -> TrainRequest:
    if payload.train_config is not None:
        return payload.train_config

    if STATE.train_config is not None:
        return STATE.train_config

    saved = _load_api_state()
    saved_config = saved.get("train_config")
    if isinstance(saved_config, dict):
        return TrainRequest(**saved_config)

    return TrainRequest(start_date=_default_start_date(), end_date=datetime.now().strftime("%Y-%m-%d"))


def _predict_from_persisted_models(payload: PredictRequest) -> Dict[str, Any]:
    base_config = _resolve_base_config(payload)
    model, return_model, multiclass_model = _load_persisted_models()

    start_date = base_config.start_date or _default_start_date()
    end_date = datetime.now().strftime("%Y-%m-%d")
    dataset = build_training_dataset(symbol=base_config.symbol, start_date=start_date, end_date=end_date)
    final_dataset, _, _ = prepare_training_matrices(dataset, drop_date=False)
    if final_dataset.empty:
        raise HTTPException(status_code=500, detail="Unable to build feature dataset for persisted inference.")

    selected_features = model.get("selected_features") if isinstance(model, dict) else None
    feature_columns = selected_features if selected_features is not None else [
        column for column in final_dataset.columns if column not in {"Date", "Target", "TargetReturn", "TargetClass"}
    ]
    feature_columns = [column for column in feature_columns if column in final_dataset.columns]
    if not feature_columns:
        raise HTTPException(status_code=500, detail="No compatible feature columns for persisted model inference.")

    latest_features = final_dataset[feature_columns].tail(1)
    latest_probability = float(predict_probability(model, latest_features)[0])
    latest_direction = int(predict_latest_direction(model, final_dataset, feature_columns))
    latest_expected_return = float(return_model.predict(latest_features)[0])
    latest_multiclass = int(multiclass_model.predict(latest_features)[0])
    latest_regime = MULTICLASS_LABELS.get(latest_multiclass, "Neutral")

    latest_row = final_dataset.tail(1)
    latest_close = float(latest_row["Close"].iloc[0])
    latest_atr = float(latest_row["ATR_14"].iloc[0]) if "ATR_14" in latest_row.columns else None
    trade_signal, stop_loss, take_profit = _compute_trade_levels(
        prob_up=latest_probability,
        current_price=latest_close,
        atr_14=latest_atr,
        buy_threshold=base_config.buy_threshold,
        sell_threshold=base_config.sell_threshold,
        sl_stop=base_config.sl_stop,
        tp_stop=base_config.tp_stop,
    )

    saved = _load_api_state()
    trained_at = saved.get("trained_at") if isinstance(saved, dict) else None

    return {
        "status": "ok",
        "trained_at": trained_at,
        "source": "persisted",
        "prediction": {
            "label": "UP" if latest_direction == 1 else "DOWN",
            "direction": latest_direction,
            "probability_increase": latest_probability,
            "expected_return": latest_expected_return,
            "regime": latest_regime,
            "trade_signal": trade_signal,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        },
    }


def _build_train_response(result: Any, request: TrainRequest) -> Dict[str, Any]:
    return {
        "status": "ok",
        "symbol": request.symbol,
        "trained_at": STATE.trained_at,
        "model_family": request.model_family,
        "metrics": result.get("metrics", {}),
        "latest": {
            "probability_increase": float(result.get("latest_probability_increase", 0.5)),
            "expected_return": float(result.get("latest_expected_return", 0.0)),
            "regime": str(result.get("latest_multiclass_label", "Neutral")),
            "trade_signal": str(result.get("trade_signal", "HOLD")),
            "stop_loss": result.get("stop_loss"),
            "take_profit": result.get("take_profit"),
        },
        "feature_count": len(result.get("feature_columns", [])),
        "rows": int(len(result.get("dataset", []))),
    }


@app.get("/")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "stock-ai-api"}


@app.post("/train")
def train_endpoint(payload: TrainRequest) -> Dict[str, Any]:
    if payload.sell_threshold >= payload.buy_threshold:
        raise HTTPException(status_code=400, detail="sell_threshold must be less than buy_threshold")

    try:
        result = train_stock_model(
            symbol=payload.symbol,
            start_date=payload.start_date,
            end_date=payload.end_date,
            buy_threshold=payload.buy_threshold,
            sell_threshold=payload.sell_threshold,
            sl_stop=payload.sl_stop,
            tp_stop=payload.tp_stop,
            model_family=payload.model_family,
            save_models=payload.save_models,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}") from exc

    STATE.result = result
    STATE.train_config = payload
    STATE.trained_at = datetime.now().isoformat(timespec="seconds")
    STATE.metrics = result.get("metrics", {})
    _save_api_state()

    return _build_train_response(result=result, request=payload)


@app.post("/predict")
def predict_endpoint(payload: PredictRequest) -> Dict[str, Any]:
    if STATE.result is None:
        if payload.retrain_if_missing:
            config = payload.train_config or TrainRequest(start_date=_default_start_date())
            train_endpoint(config)
        else:
            return _predict_from_persisted_models(payload)

    if STATE.result is None:
        raise HTTPException(status_code=500, detail="Prediction unavailable: model state missing after training")

    result = STATE.result
    dataset = result["dataset"]
    feature_columns = result["feature_columns"]
    model = result["model"]

    latest_direction = int(predict_latest_direction(model, dataset, feature_columns))
    latest_features = dataset[feature_columns].tail(1)
    latest_probability = float(predict_probability(model, latest_features)[0])

    return {
        "status": "ok",
        "trained_at": STATE.trained_at,
        "source": "memory",
        "prediction": {
            "label": "UP" if latest_direction == 1 else "DOWN",
            "direction": latest_direction,
            "probability_increase": latest_probability,
            "expected_return": float(result.get("latest_expected_return", 0.0)),
            "regime": str(result.get("latest_multiclass_label", "Neutral")),
            "trade_signal": str(result.get("trade_signal", "HOLD")),
            "stop_loss": result.get("stop_loss"),
            "take_profit": result.get("take_profit"),
        },
    }


@app.get("/metrics")
def metrics_endpoint() -> Dict[str, Any]:
    if STATE.result is not None:
        metrics = STATE.result.get("metrics", {})
        return {
            "status": "ok",
            "trained_at": STATE.trained_at,
            "source": "memory",
            "train_config": STATE.train_config.model_dump() if STATE.train_config else None,
            "metrics": metrics,
        }

    saved = _load_api_state()
    saved_metrics = saved.get("metrics") if isinstance(saved, dict) else None
    if not isinstance(saved_metrics, dict) or not saved_metrics:
        raise HTTPException(status_code=400, detail="No metrics available yet. Call /train first.")

    return {
        "status": "ok",
        "trained_at": saved.get("trained_at"),
        "source": "persisted",
        "train_config": saved.get("train_config"),
        "metrics": saved_metrics,
    }
