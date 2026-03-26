# This project is an advanced stock market prediction system
# It uses historical stock data, technical indicators, and news sentiment
# The goal is to predict whether stock price will go UP or DOWN
# Follow clean coding practices with modular, reusable functions
# Use pandas, numpy, sklearn, yfinance, ta, and transformers where needed
# Write production-level Python code with proper function definitions and docstrings

# Create a module for training and using machine learning models for stock prediction
#
# Requirements:
# - Use sklearn RandomForestClassifier as baseline model
# - Write functions:
#     train_model(X, y)
#     predict(model, X)
#
# Training:
# - Split dataset into train/test sets
# - Train model on training data
# - Return trained model
#
# Prediction:
# - Take trained model and feature data
# - Return predicted labels (0 or 1)
#
# Additional requirements:
# - Include evaluation metrics:
#     accuracy, precision, recall
# - Print model performance
# - Add feature importance visualization (optional)
# - Use clean modular code with docstrings

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence, Tuple
import importlib
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from src.transformer_model import (
        predict_transformer,
        predict_transformer_probability,
        train_transformer_model,
    )
except ImportError:
    from transformer_model import (
        predict_transformer,
        predict_transformer_probability,
        train_transformer_model,
    )


logger = logging.getLogger(__name__)
SENTIMENT_FEATURE_COLUMNS: set[str] = set()


def _sanitize_feature_frame(X: pd.DataFrame) -> pd.DataFrame:
    """Sanitize feature matrix to avoid inf/overflow failures in estimators.

    Steps:
    1. Coerce columns to numeric (invalid parsing -> NaN).
    2. Replace +/-inf with NaN.
    3. Fill NaN with median per column (fallback 0.0).
    4. Clip extreme values to a safe float64 range.
    """
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

    # Keep a wide but safe range to avoid float overflow while preserving scale.
    frame = frame.clip(lower=-1e12, upper=1e12)
    return frame


def _get_xgb_classifier() -> Any:
    """Return XGBClassifier class when xgboost is installed, else None."""
    try:
        xgboost_module = importlib.import_module("xgboost")
        return getattr(xgboost_module, "XGBClassifier", None)
    except Exception:
        return None


def _get_lgbm_classifier() -> Any:
    """Return LGBMClassifier class when lightgbm is installed, else None."""
    try:
        lightgbm_module = importlib.import_module("lightgbm")
        return getattr(lightgbm_module, "LGBMClassifier", None)
    except Exception:
        return None


def _build_rf_classifier(random_state: int) -> RandomForestClassifier:
    """Create baseline RandomForest classifier."""
    return RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)


def _build_xgb_classifier(random_state: int) -> Any:
    """Create XGBoost classifier when available, else None."""
    xgb_classifier = _get_xgb_classifier()
    if xgb_classifier is None:
        return None
    return xgb_classifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )


def _build_lgbm_classifier(random_state: int) -> Any:
    """Create LightGBM classifier when available, else None."""
    lgbm_classifier = _get_lgbm_classifier()
    if lgbm_classifier is None:
        return None
    return lgbm_classifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary",
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )


def _wrap_with_optional_scaler(estimator: Any, scale_features: bool) -> Any:
    """Optionally wrap estimator in a scaler pipeline."""
    if not scale_features:
        return estimator
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", estimator),
        ]
    )


def _predict_positive_probability(model: Any, X) -> np.ndarray:
    """Return positive-class probabilities for binary prediction."""
    X_input = _sanitize_feature_frame(pd.DataFrame(X))

    if hasattr(model, "predict_proba"):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names, but .* was fitted with feature names",
                category=UserWarning,
            )
            proba = model.predict_proba(X_input)
        return np.asarray(proba)[:, 1]

    # Fallback for estimators exposing decision_function only.
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X_input), dtype=float)
        return 1.0 / (1.0 + np.exp(-scores))

    # Last resort fallback to hard labels (keeps API robust if estimator is limited).
    labels = np.asarray(model.predict(X_input), dtype=float)
    return labels


def _predict_ensemble_labels(models: Dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    """Predict binary labels from ensemble members via soft-voting probabilities."""
    final_pred = _soft_vote_probabilities(models=models, X=X)
    return (final_pred >= 0.5).astype(int)


def _soft_vote_probabilities(models: Dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    """Return soft-voting probability by averaging member positive-class probabilities."""
    member_probs = [_predict_positive_probability(member, X) for member in models.values()]
    return np.mean(np.vstack(member_probs), axis=0)


def _extract_shap_array(shap_values: Any) -> np.ndarray:
    """Normalize SHAP output variants to a 2D (n_samples, n_features) array."""
    if isinstance(shap_values, list):
        # Binary classifier often returns [class0, class1]
        if len(shap_values) >= 2:
            return np.asarray(shap_values[1], dtype=float)
        return np.asarray(shap_values[0], dtype=float)

    arr = np.asarray(shap_values, dtype=float)
    if arr.ndim == 3:
        # Some explainers return (n_samples, n_features, n_classes)
        return arr[:, :, 1] if arr.shape[2] > 1 else arr[:, :, 0]
    return arr


def _mean_abs_shap_for_member(member: Any, X_eval: pd.DataFrame) -> np.ndarray | None:
    """Compute mean absolute SHAP values for one trained tree member."""
    estimator = member
    X_input: Any = X_eval

    if isinstance(member, Pipeline):
        scaler = member.named_steps.get("scaler")
        estimator = member.named_steps.get("classifier")
        X_input = scaler.transform(X_eval) if scaler is not None else X_eval

    if estimator is None:
        return None

    try:
        import shap  # type: ignore

        explainer = shap.TreeExplainer(estimator)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="LightGBM binary classifier with TreeExplainer shap values output has changed to a list of ndarray",
                category=UserWarning,
            )
            shap_values = explainer.shap_values(X_input)
        shap_array = _extract_shap_array(shap_values)
        if shap_array.ndim != 2:
            return None
        return np.mean(np.abs(shap_array), axis=0)
    except Exception:
        return None


def _compute_shap_importance(models: Dict[str, Any], X_eval: pd.DataFrame) -> np.ndarray:
    """Average mean-absolute SHAP importance across ensemble members."""
    collected: List[np.ndarray] = []
    for member in models.values():
        values = _mean_abs_shap_for_member(member, X_eval)
        if values is not None and values.shape[0] == X_eval.shape[1]:
            collected.append(values)

    if not collected:
        return np.zeros(X_eval.shape[1], dtype=float)
    return np.mean(np.vstack(collected), axis=0)


def _compute_permutation_importance(
    models: Dict[str, Any],
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    random_state: int,
) -> np.ndarray:
    """Compute ensemble permutation importance on an evaluation set."""
    rng = np.random.default_rng(random_state)
    base_pred = _predict_ensemble_labels(models, X_eval)
    base_score = float(accuracy_score(y_eval, base_pred))

    importances = np.zeros(X_eval.shape[1], dtype=float)
    for idx, column in enumerate(X_eval.columns):
        permuted = X_eval.copy()
        permuted[column] = rng.permutation(permuted[column].to_numpy())
        perm_pred = _predict_ensemble_labels(models, permuted)
        perm_score = float(accuracy_score(y_eval, perm_pred))
        importances[idx] = max(0.0, base_score - perm_score)

    return importances


def _normalize_importance(values: np.ndarray) -> np.ndarray:
    """Normalize non-negative importance vector into [0, 1]."""
    if values.size == 0:
        return values
    max_val = float(np.max(values))
    if max_val <= 0.0:
        return np.zeros_like(values)
    return values / max_val


def _rank_feature_scores(
    feature_columns: Sequence[str],
    shap_scores: np.ndarray,
    permutation_scores: np.ndarray,
) -> List[Tuple[str, float]]:
    """Build combined ranking from SHAP and permutation importance."""
    shap_norm = _normalize_importance(shap_scores)
    perm_norm = _normalize_importance(permutation_scores)
    combined = 0.5 * shap_norm + 0.5 * perm_norm
    ranking = [(str(name), float(score)) for name, score in zip(feature_columns, combined)]
    return sorted(ranking, key=lambda item: item[1], reverse=True)


def _select_top_features(ranking: Sequence[Tuple[str, float]], total_features: int) -> List[str]:
    """Select top features to reduce noise and overfitting."""
    if total_features <= 12:
        return [name for name, _ in ranking]
    top_k = max(12, int(round(total_features * 0.6)))
    return [name for name, _ in ranking[:top_k]]


def _print_dual_importance(
    feature_columns: Sequence[str],
    shap_scores: np.ndarray,
    permutation_scores: np.ndarray,
    ranking: Sequence[Tuple[str, float]],
    selected_features: Sequence[str],
) -> None:
    """Print SHAP, permutation, and combined selection outputs."""
    print("SHAP Feature Importance:")
    shap_pairs = sorted(
        [(str(name), float(score)) for name, score in zip(feature_columns, shap_scores)],
        key=lambda item: item[1],
        reverse=True,
    )
    for idx, (name, score) in enumerate(shap_pairs, start=1):
        print(f"{idx:>2}. {name:<24} {score:.6f}")

    print("Permutation Feature Importance:")
    perm_pairs = sorted(
        [(str(name), float(score)) for name, score in zip(feature_columns, permutation_scores)],
        key=lambda item: item[1],
        reverse=True,
    )
    for idx, (name, score) in enumerate(perm_pairs, start=1):
        print(f"{idx:>2}. {name:<24} {score:.6f}")

    print("Combined Feature Ranking (SHAP + Permutation):")
    for idx, (name, score) in enumerate(ranking, start=1):
        print(f"{idx:>2}. {name:<24} {score:.6f}")

    print(f"Selected top features ({len(selected_features)}): {list(selected_features)}")


def _average_feature_importances(models: Dict[str, Any], feature_columns: Sequence[str]) -> List[Tuple[str, float]]:
    """Average available feature importances across ensemble members."""
    collected: List[np.ndarray] = []
    for member in models.values():
        estimator = _get_fitted_estimator(member)
        if hasattr(estimator, "feature_importances_"):
            values = np.asarray(getattr(estimator, "feature_importances_"), dtype=float)
            if values.shape[0] == len(feature_columns):
                collected.append(values)

    if not collected:
        return []

    mean_importance = np.mean(np.vstack(collected), axis=0)
    pairs = [(str(name), float(score)) for name, score in zip(feature_columns, mean_importance)]
    return sorted(pairs, key=lambda item: item[1], reverse=True)


def _build_ensemble_members(random_state: int, scale_features: bool) -> Dict[str, Any]:
    """Create ensemble members with optional leakage-safe scaling wrappers."""
    rf_model = _wrap_with_optional_scaler(_build_rf_classifier(random_state=random_state), scale_features)
    xgb_base = _build_xgb_classifier(random_state=random_state)
    lgbm_base = _build_lgbm_classifier(random_state=random_state)

    if xgb_base is None:
        logger.warning("xgboost is not installed. Ensemble will exclude XGBoost.")
    if lgbm_base is None:
        logger.warning("lightgbm is not installed. Ensemble will exclude LightGBM.")

    members: Dict[str, Any] = {"rf": rf_model}
    if xgb_base is not None:
        members["xgb"] = _wrap_with_optional_scaler(xgb_base, scale_features)
    if lgbm_base is not None:
        members["lgbm"] = _wrap_with_optional_scaler(lgbm_base, scale_features)
    return members


def _build_single_classifier(model_family: str, random_state: int, scale_features: bool) -> Any:
    """Build one classifier based on requested model family with safe fallbacks."""
    family = str(model_family).lower()

    if family == "xgboost":
        estimator = _build_xgb_classifier(random_state=random_state)
        if estimator is None:
            logger.warning("xgboost unavailable. Falling back to LightGBM/RandomForest.")
            estimator = _build_lgbm_classifier(random_state=random_state) or _build_rf_classifier(random_state=random_state)
        return _wrap_with_optional_scaler(estimator, scale_features)

    if family == "lightgbm":
        estimator = _build_lgbm_classifier(random_state=random_state)
        if estimator is None:
            logger.warning("lightgbm unavailable. Falling back to XGBoost/RandomForest.")
            estimator = _build_xgb_classifier(random_state=random_state) or _build_rf_classifier(random_state=random_state)
        return _wrap_with_optional_scaler(estimator, scale_features)

    if family == "random_forest":
        return _wrap_with_optional_scaler(_build_rf_classifier(random_state=random_state), scale_features)

    raise ValueError("Unsupported model_family for single-model training.")


def train_soft_voting_ensemble(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
    scale_features: bool = True,
) -> Dict[str, Any]:
    """Train RandomForest, XGBoost, and LightGBM and return soft-voting model bundle."""
    X_df = pd.DataFrame(X).reset_index(drop=True)
    y_series = pd.Series(y).reset_index(drop=True)
    if len(X_df) != len(y_series):
        raise ValueError("X and y must have the same number of rows.")

    members = _build_ensemble_members(random_state=random_state, scale_features=scale_features)
    for name, member in members.items():
        member.fit(X_df, y_series)
        logger.info("Trained ensemble member: %s", name)

    return {
        "type": "ensemble",
        "voting": "soft",
        "models": members,
        "member_names": list(members.keys()),
        "selected_features": list(X_df.columns),
    }


def _walk_forward_year_splits(dates: pd.Series) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    """Build expanding-window year splits: train on past years, test on next year."""
    dt = pd.to_datetime(dates)
    years = dt.dt.year
    unique_years = sorted(int(year) for year in years.dropna().unique().tolist())
    folds: List[Tuple[np.ndarray, np.ndarray, int]] = []

    for test_year in unique_years[1:]:
        train_idx = np.where((years < test_year).to_numpy())[0]
        test_idx = np.where((years == test_year).to_numpy())[0]
        if train_idx.size == 0 or test_idx.size == 0:
            continue
        folds.append((train_idx, test_idx, test_year))

    return folds


def _get_fitted_estimator(model: Any) -> Any:
    """Return fitted estimator, unwrapping sklearn Pipeline when needed."""
    if isinstance(model, Pipeline):
        return model.named_steps["classifier"]
    return model


def _compute_feature_importance_ranking(model: Any, feature_columns: Sequence[str]) -> List[Tuple[str, float]]:
    """Compute descending (feature, importance) ranking when model supports it."""
    if isinstance(model, dict) and model.get("type") == "ensemble" and "models" in model:
        return _average_feature_importances(model["models"], feature_columns)

    if isinstance(model, dict) and model.get("type") == "single" and "estimator" in model:
        estimator = _get_fitted_estimator(model["estimator"])
        if not hasattr(estimator, "feature_importances_"):
            return []
        importances = getattr(estimator, "feature_importances_")
        pairs = [(str(name), float(score)) for name, score in zip(feature_columns, importances)]
        return sorted(pairs, key=lambda item: item[1], reverse=True)

    estimator = _get_fitted_estimator(model)
    if not hasattr(estimator, "feature_importances_"):
        return []

    importances = getattr(estimator, "feature_importances_")
    pairs = [(str(name), float(score)) for name, score in zip(feature_columns, importances)]
    return sorted(pairs, key=lambda item: item[1], reverse=True)


def _print_importance_ranking(ranking: Sequence[Tuple[str, float]]) -> None:
    """Print global and sentiment-specific feature-importance rankings."""
    if not ranking:
        print("Feature importance is unavailable for the selected model.")
        return

    print("Feature Importance Ranking:")
    for idx, (name, score) in enumerate(ranking, start=1):
        print(f"{idx:>2}. {name:<24} {score:.6f}")


def train_model(
    X,
    y,
    random_state: int = 42,
    scale_features: bool = True,
    feature_columns: Sequence[str] | None = None,
    dates: pd.Series | None = None,
    use_feature_selection: bool = False,
    model_family: str = "ensemble",
):
    requested_family = str(model_family).lower()

    if requested_family in {"informer", "tft"}:
        selected_columns = list(feature_columns) if feature_columns is not None else list(pd.DataFrame(X).columns)
        transformer_model, metrics = train_transformer_model(
            X=pd.DataFrame(X)[selected_columns],
            y=pd.Series(y),
            feature_columns=selected_columns,
            dates=dates,
            model_family=requested_family,
        )
        print(
            "Model Performance:\n"
            f"Accuracy: {metrics['accuracy']:.3f}\n"
            f"Precision: {metrics['precision']:.3f}\n"
            f"Recall: {metrics['recall']:.3f}"
        )
        print(f"Transformer model family: {requested_family}")
        return transformer_model, metrics

    """Train RF/XGB/LGBM ensemble and return model bundle plus evaluation metrics."""
    X_df = _sanitize_feature_frame(pd.DataFrame(X)).reset_index(drop=True)
    y_series = pd.Series(y).reset_index(drop=True)

    if len(X_df) != len(y_series):
        raise ValueError("X and y must have the same number of rows.")

    if dates is not None and len(dates) == len(X_df):
        folds = _walk_forward_year_splits(pd.Series(dates).reset_index(drop=True))
    else:
        folds = []

    if not folds:
        # Fallback to chronology-preserving splits when year-based folds are unavailable.
        n_rows = len(X_df)
        n_splits = max(2, min(5, n_rows - 1))
        splitter = TimeSeriesSplit(n_splits=n_splits)
        folds = [(train_idx, test_idx, -1) for train_idx, test_idx in splitter.split(X_df)]

    oof_true: List[int] = []
    oof_pred: List[int] = []
    fold_shap_scores: List[np.ndarray] = []
    fold_perm_scores: List[np.ndarray] = []

    fold_importance_scores: List[np.ndarray] = []

    for train_idx, test_idx, test_year in folds:
        X_train, X_test = X_df.iloc[train_idx], X_df.iloc[test_idx]
        X_train = _sanitize_feature_frame(X_train)
        X_test = _sanitize_feature_frame(X_test)
        y_train, y_test = y_series.iloc[train_idx], y_series.iloc[test_idx]

        if requested_family == "ensemble":
            fold_members = _build_ensemble_members(random_state=random_state, scale_features=scale_features)
            for name, member in fold_members.items():
                member.fit(X_train, y_train)
                logger.info("Trained fold member: %s", name)

            # Soft voting: average member probabilities and then threshold for class.
            final_pred = _soft_vote_probabilities(models=fold_members, X=X_test)
            fold_pred = (final_pred >= 0.5).astype(int)

            if use_feature_selection:
                shap_scores = _compute_shap_importance(fold_members, X_test)
                perm_scores = _compute_permutation_importance(
                    fold_members,
                    X_test,
                    y_test,
                    random_state=random_state,
                )
                fold_shap_scores.append(shap_scores)
                fold_perm_scores.append(perm_scores)
        else:
            fold_model = _build_single_classifier(
                model_family=requested_family,
                random_state=random_state,
                scale_features=scale_features,
            )
            fold_model.fit(X_train, y_train)
            fold_proba = _predict_positive_probability(fold_model, X_test)
            fold_pred = (fold_proba >= 0.5).astype(int)

            estimator = _get_fitted_estimator(fold_model)
            if hasattr(estimator, "feature_importances_"):
                values = np.asarray(getattr(estimator, "feature_importances_"), dtype=float)
                if values.shape[0] == X_df.shape[1]:
                    fold_importance_scores.append(values)

        oof_true.extend(y_test.astype(int).tolist())
        oof_pred.extend(fold_pred.astype(int).tolist())

        if test_year != -1:
            logger.info("Completed walk-forward fold with test year: %s", test_year)

    y_eval_true = np.asarray(oof_true, dtype=int)
    y_eval_pred = np.asarray(oof_pred, dtype=int)

    metrics = {
        "accuracy": float(accuracy_score(y_eval_true, y_eval_pred)),
        "precision": float(precision_score(y_eval_true, y_eval_pred, zero_division=0)),
        "recall": float(recall_score(y_eval_true, y_eval_pred, zero_division=0)),
    }

    selected_columns = list(feature_columns) if feature_columns is not None else list(X_df.columns)
    if use_feature_selection and fold_shap_scores and fold_perm_scores:
        mean_shap = np.mean(np.vstack(fold_shap_scores), axis=0)
        mean_perm = np.mean(np.vstack(fold_perm_scores), axis=0)
        combined_ranking = _rank_feature_scores(selected_columns, mean_shap, mean_perm)
        selected_columns = _select_top_features(combined_ranking, total_features=len(selected_columns))
        _print_dual_importance(
            feature_columns=list(X_df.columns),
            shap_scores=mean_shap,
            permutation_scores=mean_perm,
            ranking=combined_ranking,
            selected_features=selected_columns,
        )
    elif use_feature_selection and requested_family != "ensemble" and fold_importance_scores:
        mean_importance = np.mean(np.vstack(fold_importance_scores), axis=0)
        ranked = sorted(
            [(str(name), float(score)) for name, score in zip(list(X_df.columns), mean_importance)],
            key=lambda item: item[1],
            reverse=True,
        )
        selected_columns = _select_top_features(ranked, total_features=len(selected_columns))
        print(f"Selected top features ({len(selected_columns)}): {list(selected_columns)}")

    # Train final model on all data for downstream latest prediction.
    if requested_family == "ensemble":
        ensemble_members = _build_ensemble_members(random_state=random_state, scale_features=scale_features)
        X_full = _sanitize_feature_frame(X_df[selected_columns])
        for name, member in ensemble_members.items():
            member.fit(X_full, y_series)
            logger.info("Trained final ensemble member on full data: %s", name)
        model_bundle: Dict[str, Any] = {
            "type": "ensemble",
            "voting": "soft",
            "models": ensemble_members,
            "member_names": list(ensemble_members.keys()),
            "selected_features": selected_columns,
        }
    else:
        single_estimator = _build_single_classifier(
            model_family=requested_family,
            random_state=random_state,
            scale_features=scale_features,
        )
        X_full = _sanitize_feature_frame(X_df[selected_columns])
        single_estimator.fit(X_full, y_series)
        model_bundle = {
            "type": "single",
            "model_family": requested_family,
            "estimator": single_estimator,
            "selected_features": selected_columns,
        }

    print(
        "Model Performance:\n"
        f"Accuracy: {metrics['accuracy']:.3f}\n"
        f"Precision: {metrics['precision']:.3f}\n"
        f"Recall: {metrics['recall']:.3f}"
    )

    ranking = _compute_feature_importance_ranking(model_bundle, selected_columns)
    _print_importance_ranking(ranking)

    return model_bundle, metrics


def predict(model: Any, X):
    """Return predicted binary labels (0 or 1) for provided feature rows."""
    if isinstance(model, dict) and model.get("type") == "transformer":
        return predict_transformer(model, pd.DataFrame(X))
    if isinstance(model, dict) and model.get("type") == "ensemble" and "models" in model:
        selected_features = model.get("selected_features")
        X_input = X[selected_features] if selected_features is not None else X
        X_input = _sanitize_feature_frame(pd.DataFrame(X_input))
        final_pred = _soft_vote_probabilities(models=model["models"], X=X_input)
        return (final_pred >= 0.5).astype(int)
    if isinstance(model, dict) and model.get("type") == "single" and "estimator" in model:
        selected_features = model.get("selected_features")
        X_input = X[selected_features] if selected_features is not None else X
        X_input = _sanitize_feature_frame(pd.DataFrame(X_input))
        probs = _predict_positive_probability(model["estimator"], X_input)
        return (probs >= 0.5).astype(int)
    if isinstance(model, dict):
        raise ValueError("Unsupported model dictionary format for prediction.")
    X_input = _sanitize_feature_frame(pd.DataFrame(X))
    return model.predict(X_input)


def predict_probability(model: Any, X) -> np.ndarray:
    """Return probability of price increase for provided feature rows."""
    if isinstance(model, dict) and model.get("type") == "transformer":
        return predict_transformer_probability(model, pd.DataFrame(X))
    if isinstance(model, dict) and model.get("type") == "ensemble" and "models" in model:
        selected_features = model.get("selected_features")
        X_input = X[selected_features] if selected_features is not None else X
        X_input = _sanitize_feature_frame(pd.DataFrame(X_input))
        return _soft_vote_probabilities(models=model["models"], X=X_input)
    if isinstance(model, dict) and model.get("type") == "single" and "estimator" in model:
        selected_features = model.get("selected_features")
        X_input = X[selected_features] if selected_features is not None else X
        X_input = _sanitize_feature_frame(pd.DataFrame(X_input))
        return _predict_positive_probability(model["estimator"], X_input)
    if isinstance(model, dict):
        raise ValueError("Unsupported model dictionary format for probability prediction.")
    X_input = _sanitize_feature_frame(pd.DataFrame(X))
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_input)
        return np.asarray(proba)[:, 1]
    labels = np.asarray(model.predict(X_input), dtype=float)
    return labels


def plot_feature_importance(model: Any, feature_names: Sequence[str]) -> None:
    """Visualize feature importances for a trained tree-based model or pipeline."""
    if isinstance(model, dict) and model.get("type") == "ensemble" and "models" in model:
        ranking = _average_feature_importances(model["models"], feature_names)
        if not ranking:
            raise ValueError("Provided ensemble model does not expose feature importances.")
        sorted_labels = [name for name, _ in ranking]
        sorted_values = [score for _, score in ranking]

        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances (Ensemble Average)")
        plt.bar(range(len(sorted_labels)), sorted_values, align="center")
        plt.xticks(range(len(sorted_labels)), sorted_labels, rotation=75)
        plt.tight_layout()
        plt.show()
        return

    estimator = _get_fitted_estimator(model)
    if not hasattr(estimator, "feature_importances_"):
        raise ValueError("Provided model does not expose feature_importances_.")

    importances = estimator.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names_list = list(feature_names)
    sorted_labels = [feature_names_list[int(i)] for i in indices]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(feature_names_list)), importances[indices], align="center")
    plt.xticks(range(len(feature_names_list)), sorted_labels, rotation=75)
    plt.tight_layout()
    plt.show()


def time_series_cv_accuracy(X, y, n_splits: int = 5, random_state: int = 42) -> float:
    """Evaluate average accuracy using TimeSeriesSplit.

    Random train/test shuffling leaks future patterns into training for financial
    data, which can overstate performance. TimeSeriesSplit preserves chronology:
    each fold trains on past windows and validates on future windows.
    """
    splitter = TimeSeriesSplit(n_splits=n_splits)
    fold_scores: List[float] = []
    xgb_classifier = _get_xgb_classifier()

    X_sanitized = _sanitize_feature_frame(pd.DataFrame(X))

    for train_idx, test_idx in splitter.split(X_sanitized):
        X_train, X_test = X_sanitized.iloc[train_idx], X_sanitized.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if xgb_classifier is not None:
            fold_model = xgb_classifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=-1,
            )
        else:
            fold_model = RandomForestClassifier(n_estimators=250, random_state=random_state, n_jobs=-1)

        fold_model.fit(X_train, y_train)
        predictions = fold_model.predict(X_test)
        fold_scores.append(float(accuracy_score(y_test, predictions)))

    avg_accuracy = float(np.mean(fold_scores)) if fold_scores else 0.0
    print(f"TimeSeriesSplit average accuracy ({n_splits} folds): {avg_accuracy:.3f}")
    return avg_accuracy

    