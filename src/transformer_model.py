from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        pe_buffer = getattr(self, "pe")
        return x + pe_buffer[:, :seq_len, :]


class TransformerTSClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model=d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        pooled = x[:, -1, :]
        logits = self.head(pooled).squeeze(-1)
        return logits


def _build_folds(dates: pd.Series | None, n_rows: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    if dates is None or len(dates) != n_rows:
        split_idx = int(n_rows * 0.8)
        train_idx = np.arange(0, split_idx)
        test_idx = np.arange(split_idx, n_rows)
        return [(train_idx, test_idx)]

    years = pd.to_datetime(dates).dt.year
    unique_years = sorted(int(y) for y in years.dropna().unique().tolist())
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for test_year in unique_years[1:]:
        train_idx = np.where((years < test_year).to_numpy())[0]
        test_idx = np.where((years == test_year).to_numpy())[0]
        if train_idx.size == 0 or test_idx.size == 0:
            continue
        folds.append((train_idx, test_idx))

    if not folds:
        split_idx = int(n_rows * 0.8)
        folds.append((np.arange(0, split_idx), np.arange(split_idx, n_rows)))
    return folds


def _make_sequences_by_end_indices(
    X_scaled: np.ndarray,
    y: np.ndarray,
    end_indices: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seqs: List[np.ndarray] = []
    labels: List[int] = []
    valid_end_idx: List[int] = []

    for end_idx in end_indices:
        if end_idx < seq_len - 1:
            continue
        start = end_idx - seq_len + 1
        seqs.append(X_scaled[start : end_idx + 1])
        labels.append(int(y[end_idx]))
        valid_end_idx.append(int(end_idx))

    if not seqs:
        return (
            np.zeros((0, seq_len, X_scaled.shape[1]), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    return np.asarray(seqs, dtype=np.float32), np.asarray(labels, dtype=np.float32), np.asarray(valid_end_idx, dtype=np.int64)


def _train_one_model(
    X_train_seq: np.ndarray,
    y_train_seq: np.ndarray,
    input_dim: int,
    model_family: str,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> TransformerTSClassifier:
    if model_family == "tft":
        d_model, nhead, num_layers, dropout = 64, 4, 3, 0.2
    else:
        d_model, nhead, num_layers, dropout = 64, 4, 2, 0.1

    model = TransformerTSClassifier(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    X_tensor = torch.tensor(X_train_seq, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y_train_seq, dtype=torch.float32, device=device)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(X_tensor)
        loss = criterion(logits, y_tensor)
        loss.backward()
        optimizer.step()

    return model


def _predict_probs_on_sequences(model: TransformerTSClassifier, X_seq: np.ndarray, device: torch.device) -> np.ndarray:
    if X_seq.shape[0] == 0:
        return np.zeros((0,), dtype=float)

    model.eval()
    with torch.no_grad():
        x = torch.tensor(X_seq, dtype=torch.float32, device=device)
        logits = model(x)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
    return probs.astype(float)


def train_transformer_model(
    X: pd.DataFrame,
    y: pd.Series,
    feature_columns: Sequence[str],
    dates: pd.Series | None,
    model_family: str = "informer",
    seq_len: int = 20,
    epochs: int = 12,
    learning_rate: float = 1e-3,
    random_state: int = 42,
) -> tuple[Dict[str, Any], Dict[str, float]]:
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    X_df = pd.DataFrame(X).reset_index(drop=True)
    y_series = pd.Series(y).astype(int).reset_index(drop=True)
    X_np = X_df.to_numpy(dtype=np.float32)
    y_np = y_series.to_numpy(dtype=np.int64)

    folds = _build_folds(dates.reset_index(drop=True) if dates is not None else None, len(X_df))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_true: List[int] = []
    all_pred: List[int] = []

    for train_idx, test_idx in folds:
        scaler = StandardScaler()
        scaler.fit(X_np[train_idx])
        X_scaled = scaler.transform(X_np)

        X_train_seq, y_train_seq, _ = _make_sequences_by_end_indices(X_scaled, y_np, train_idx, seq_len)
        X_test_seq, y_test_seq, _ = _make_sequences_by_end_indices(X_scaled, y_np, test_idx, seq_len)

        if X_train_seq.shape[0] == 0 or X_test_seq.shape[0] == 0:
            continue

        fold_model = _train_one_model(
            X_train_seq=X_train_seq,
            y_train_seq=y_train_seq,
            input_dim=X_scaled.shape[1],
            model_family=model_family,
            epochs=epochs,
            learning_rate=learning_rate,
            device=device,
        )

        test_probs = _predict_probs_on_sequences(fold_model, X_test_seq, device=device)
        test_pred = (test_probs >= 0.5).astype(int)

        all_true.extend(y_test_seq.astype(int).tolist())
        all_pred.extend(test_pred.astype(int).tolist())

    if not all_true:
        metrics = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0}
    else:
        y_true = np.asarray(all_true, dtype=int)
        y_hat = np.asarray(all_pred, dtype=int)
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_hat)),
            "precision": float(precision_score(y_true, y_hat, zero_division=0)),
            "recall": float(recall_score(y_true, y_hat, zero_division=0)),
        }

    final_scaler = StandardScaler()
    X_scaled_full = final_scaler.fit_transform(X_np)
    all_idx = np.arange(len(X_df))
    X_full_seq, y_full_seq, _ = _make_sequences_by_end_indices(X_scaled_full, y_np, all_idx, seq_len)

    final_model = _train_one_model(
        X_train_seq=X_full_seq,
        y_train_seq=y_full_seq,
        input_dim=X_scaled_full.shape[1],
        model_family=model_family,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
    )

    bundle: Dict[str, Any] = {
        "type": "transformer",
        "family": model_family,
        "model": final_model,
        "scaler": final_scaler,
        "seq_len": seq_len,
        "selected_features": list(feature_columns),
        "device": str(device),
    }
    return bundle, metrics


def predict_transformer_probability(model_bundle: Dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    X_df = pd.DataFrame(X)
    selected = model_bundle.get("selected_features", list(X_df.columns))
    X_input = X_df[selected].to_numpy(dtype=np.float32)

    scaler: StandardScaler = model_bundle["scaler"]
    model: TransformerTSClassifier = model_bundle["model"]
    seq_len = int(model_bundle.get("seq_len", 20))
    device = torch.device(model_bundle.get("device", "cpu"))

    X_scaled = scaler.transform(X_input)
    probs = np.full((X_scaled.shape[0],), 0.5, dtype=float)

    if X_scaled.shape[0] < seq_len:
        return probs

    seqs = []
    end_positions = []
    for end_idx in range(seq_len - 1, X_scaled.shape[0]):
        seqs.append(X_scaled[end_idx - seq_len + 1 : end_idx + 1])
        end_positions.append(end_idx)

    seq_array = np.asarray(seqs, dtype=np.float32)
    pred_probs = _predict_probs_on_sequences(model, seq_array, device=device)

    for end_idx, prob in zip(end_positions, pred_probs):
        probs[end_idx] = float(prob)

    return probs


def predict_transformer(model_bundle: Dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    probs = predict_transformer_probability(model_bundle, X)
    return (probs >= 0.5).astype(int)
