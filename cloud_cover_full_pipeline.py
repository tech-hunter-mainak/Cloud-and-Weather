
#!/usr/bin/env python3
"""
End-to-end cloud cover prediction pipeline.

Models:
- XGBoost baseline
- LSTM temporal model
- Optional simple GCN and ST-GCN (pure PyTorch, no PyG required)

Assumptions:
- CSV contains a target column named `cloud`
- A timestamp column exists: `last_updated_epoch` or `last_updated`
- Spatial columns exist: `latitude`, `longitude`
- Optional categorical columns are handled safely if present

This script is defensive:
- validates columns
- handles missing optional dependencies gracefully
- skips graph models if the data are not sufficiently synchronized across timestamps
"""

from __future__ import annotations

import os
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")


# -----------------------------
# Configuration
# -----------------------------

@dataclass
class Config:
    csv_path: str = "data/IndianWeatherRepository.csv"
    target_col: str = "cloud"
    timestamp_epoch_col: str = "last_updated_epoch"
    timestamp_text_col: str = "last_updated"
    location_col: str = "location_name"
    lat_col: str = "latitude"
    lon_col: str = "longitude"
    region_col: str = "region"
    timezone_col: str = "timezone"

    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42

    sequence_length: int = 7
    knn_k: int = 5

    output_dir: str = "outputs"

    # classification bins for cloud cover
    clear_max: float = 20.0
    partly_max: float = 60.0


CFG = Config()


# -----------------------------
# Utility helpers
# -----------------------------

def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def parse_time_to_minutes(value) -> float:
    """Parse time-like values into minutes since midnight. Returns NaN if not parseable."""
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    if not s:
        return np.nan
    # Try common time formats
    for fmt in ("%H:%M", "%H:%M:%S", "%I:%M %p", "%I:%M:%S %p"):
        try:
            t = pd.to_datetime(s, format=fmt)
            return float(t.hour * 60 + t.minute + t.second / 60.0)
        except Exception:
            pass
    # Fallback: if only hour is present
    try:
        t = pd.to_datetime(s)
        return float(t.hour * 60 + t.minute + t.second / 60.0)
    except Exception:
        return np.nan


def parse_timestamp(df: pd.DataFrame, cfg: Config) -> pd.Series:
    if cfg.timestamp_epoch_col in df.columns:
        ts = pd.to_datetime(df[cfg.timestamp_epoch_col], unit="s", errors="coerce")
        if ts.notna().any():
            return ts
    if cfg.timestamp_text_col in df.columns:
        ts = pd.to_datetime(df[cfg.timestamp_text_col], errors="coerce")
        return ts
    raise ValueError(
        f"Could not find a usable timestamp column. Expected '{cfg.timestamp_epoch_col}' or '{cfg.timestamp_text_col}'."
    )


def cloud_to_class(y: pd.Series, clear_max: float, partly_max: float) -> pd.Series:
    return pd.cut(
        y,
        bins=[-np.inf, clear_max, partly_max, np.inf],
        labels=["Clear", "Partly Cloudy", "Overcast"],
        include_lowest=True,
        right=True,
    )


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred, squared=False)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(y_true, y_pred, y_prob=None) -> Dict[str, float]:
    out = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "Recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "F1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if y_prob is not None:
        try:
            out["ROC_AUC_ovr_macro"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            )
        except Exception:
            out["ROC_AUC_ovr_macro"] = np.nan
    return out


# -----------------------------
# Data loading / preprocessing
# -----------------------------

def load_data(cfg: Config) -> pd.DataFrame:
    if not os.path.exists(cfg.csv_path):
        raise FileNotFoundError(f"CSV file not found: {cfg.csv_path}")
    df = pd.read_csv(cfg.csv_path)
    if df.empty:
        raise ValueError("Loaded dataset is empty.")
    return df


def basic_cleaning(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()

    # Normalize column names slightly while preserving original names
    df.columns = [c.strip() for c in df.columns]

    if cfg.target_col not in df.columns:
        raise ValueError(f"Target column '{cfg.target_col}' not found in the dataset.")

    # Ensure timestamps
    df["_timestamp"] = parse_timestamp(df, cfg)

    # Convert target and key numeric columns to numeric where possible
    for col in df.columns:
        if col == "_timestamp":
            continue
        if col in [
            "country", "location_name", "region", "timezone", "condition_text",
            "wind_direction", "sunrise", "sunset", "moonrise", "moonset", "moon_phase"
        ]:
            continue
        if df[col].dtype == "object":
            # Safe conversion for likely numeric object columns
            converted = pd.to_numeric(df[col], errors="ignore")
            df[col] = converted

    # Time-derived features
    df["_hour"] = df["_timestamp"].dt.hour.astype("float")
    df["_day"] = df["_timestamp"].dt.day.astype("float")
    df["_month"] = df["_timestamp"].dt.month.astype("float")
    df["_dayofweek"] = df["_timestamp"].dt.dayofweek.astype("float")
    df["_dayofyear"] = df["_timestamp"].dt.dayofyear.astype("float")
    df["_quarter"] = df["_timestamp"].dt.quarter.astype("float")

    # Season (India-friendly rough split)
    # 1: Winter, 2: Pre-monsoon, 3: Monsoon, 4: Post-monsoon
    month = df["_month"]
    season = np.select(
        [
            month.isin([12, 1, 2]),
            month.isin([3, 4, 5]),
            month.isin([6, 7, 8, 9]),
            month.isin([10, 11]),
        ],
        [1, 2, 3, 4],
        default=np.nan,
    )
    df["_season"] = season.astype("float")

    # Cyclic encodings
    for col, period in [("_hour", 24), ("_month", 12), ("_dayofweek", 7), ("_dayofyear", 365.25)]:
        df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
        df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)

    # Parse sunrise/sunset/moonrise/moonset into minutes
    for col in ["sunrise", "sunset", "moonrise", "moonset"]:
        if col in df.columns:
            df[f"{col}_minutes"] = df[col].apply(parse_time_to_minutes)

    # Parse wind direction text if useful; keep as category too
    if "wind_direction" in df.columns:
        # No conversion required; keep as categorical
        pass

    # Redundant-unit columns are kept out of model feature lists later
    return df


def make_feature_lists(df: pd.DataFrame, cfg: Config) -> Tuple[List[str], List[str], List[str]]:
    # Numeric features we expect to be useful
    numeric_candidates = [
        "temperature_celsius",
        "wind_kph",
        "pressure_mb",
        "precip_mm",
        "humidity",
        "cloud",
        "feels_like_celsius",
        "visibility_km",
        "uv_index",
        "gust_kph",
        "air_quality_Carbon_Monoxide",
        "air_quality_Ozone",
        "air_quality_Nitrogen_dioxide",
        "air_quality_Sulphur_dioxide",
        "air_quality_PM2.5",
        "air_quality_PM10",
        "air_quality_us-epa-index",
        "air_quality_gb-defra-index",
        "latitude",
        "longitude",
        "_hour",
        "_day",
        "_month",
        "_dayofweek",
        "_dayofyear",
        "_quarter",
        "_season",
        "_hour_sin",
        "_hour_cos",
        "_month_sin",
        "_month_cos",
        "_dayofweek_sin",
        "_dayofweek_cos",
        "_dayofyear_sin",
        "_dayofyear_cos",
        "sunrise_minutes",
        "sunset_minutes",
        "moonrise_minutes",
        "moonset_minutes",
        "moon_illumination",
    ]

    numeric_features = [c for c in numeric_candidates if c in df.columns and c != cfg.target_col]

    categorical_candidates = [
        "region",
        "timezone",
        "condition_text",
        "wind_direction",
        "moon_phase",
    ]
    categorical_features = [c for c in categorical_candidates if c in df.columns]

    spatial_features = [c for c in [cfg.lat_col, cfg.lon_col] if c in df.columns]
    return numeric_features, categorical_features, spatial_features


def train_val_test_split_by_time(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("_timestamp").reset_index(drop=True)
    n = len(df)
    test_n = max(1, int(n * cfg.test_size))
    val_n = max(1, int(n * cfg.val_size))
    train_n = n - test_n - val_n
    if train_n <= 0:
        raise ValueError("Dataset too small for the requested train/val/test split.")
    train_df = df.iloc[:train_n].copy()
    val_df = df.iloc[train_n:train_n + val_n].copy()
    test_df = df.iloc[train_n + val_n:].copy()
    return train_df, val_df, test_df


def make_xy(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    cfg: Config,
) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.loc[:, list(feature_cols)].copy()
    y = pd.to_numeric(df[cfg.target_col], errors="coerce")
    valid = y.notna()
    X = X.loc[valid].copy()
    y = y.loc[valid].copy()
    return X, y


# -----------------------------
# XGBoost model
# -----------------------------

def build_xgb_pipeline(numeric_features: List[str], categorical_features: List[str]):
    try:
        from xgboost import XGBRegressor
    except ImportError as e:
        raise ImportError(
            "xgboost is not installed. Install it with: pip install xgboost"
        ) from e

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=CFG.random_state,
        objective="reg:squarederror",
        tree_method="hist",
    )

    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    return pipe


def train_xgb(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
    cfg: Config,
) -> Tuple[Pipeline, Dict[str, float]]:
    feature_cols = numeric_features + categorical_features
    X_train, y_train = make_xy(train_df, feature_cols, cfg)
    X_val, y_val = make_xy(val_df, feature_cols, cfg)
    X_test, y_test = make_xy(test_df, feature_cols, cfg)

    model = build_xgb_pipeline(numeric_features, categorical_features)
    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    metrics = {
        **{f"val_{k}": v for k, v in regression_metrics(y_val, val_pred).items()},
        **{f"test_{k}": v for k, v in regression_metrics(y_test, test_pred).items()},
    }

    return model, metrics


def xgb_classification_report(pipe: Pipeline, df: pd.DataFrame, feature_cols: List[str], cfg: Config) -> Dict[str, float]:
    X, y_reg = make_xy(df, feature_cols, cfg)
    y_cls = cloud_to_class(y_reg, cfg.clear_max, cfg.partly_max)

    # Drop any rows with NaN classes
    valid = y_cls.notna()
    X = X.loc[valid].copy()
    y_cls = y_cls.loc[valid].astype(str)

    pred_reg = pipe.predict(X)
    pred_cls = cloud_to_class(pd.Series(pred_reg, index=X.index), cfg.clear_max, cfg.partly_max).astype(str)

    labels = ["Clear", "Partly Cloudy", "Overcast"]
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    y_true_idx = y_cls.map(label_to_idx)
    y_pred_idx = pred_cls.map(label_to_idx)

    # Probability-like scores are not available from regressor;
    # compute one-vs-rest AUROC only if class probabilities are explicitly derived, so skip here.
    return classification_metrics(y_true_idx, y_pred_idx)


# -----------------------------
# LSTM temporal model
# -----------------------------

def create_lstm_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    cfg: Config,
    sequence_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding sequences per location:
    x[t-seq_len:t] -> y[t]
    """
    if cfg.location_col not in df.columns:
        raise ValueError(
            f"LSTM requires '{cfg.location_col}' for per-location sequences."
        )

    # Prepare feature matrix
    work = df.copy()
    work = work.sort_values([cfg.location_col, "_timestamp"]).reset_index(drop=True)

    X_list, y_list = [], []

    for loc, grp in work.groupby(cfg.location_col, sort=False):
        grp = grp.sort_values("_timestamp").reset_index(drop=True)
        feat = grp[feature_cols].copy()
        y = pd.to_numeric(grp[cfg.target_col], errors="coerce")
        feat = feat.apply(pd.to_numeric, errors="coerce")

        # Ensure enough rows
        if len(grp) <= sequence_length:
            continue

        for i in range(sequence_length, len(grp)):
            x_seq = feat.iloc[i - sequence_length:i].to_numpy(dtype=np.float32)
            y_val = y.iloc[i]
            if np.isnan(x_seq).any() or pd.isna(y_val):
                continue
            X_list.append(x_seq)
            y_list.append(float(y_val))

    if not X_list:
        raise ValueError(
            "No valid LSTM sequences could be created. Check sequence length and missing values."
        )

    return np.stack(X_list), np.array(y_list, dtype=np.float32)


def build_lstm_model(input_shape: Tuple[int, int]):
    try:
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
    except ImportError as e:
        raise ImportError(
            "TensorFlow is not installed. Install it with: pip install tensorflow"
        ) from e

    model = Sequential(
        [
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_lstm_model(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: List[str],
    cfg: Config,
    sequence_length: int,
):
    try:
        from tensorflow.keras.callbacks import EarlyStopping
    except ImportError as e:
        raise ImportError(
            "TensorFlow is not installed. Install it with: pip install tensorflow"
        ) from e

    X_train, y_train = create_lstm_sequences(df_train, feature_cols, cfg, sequence_length)
    X_val, y_val = create_lstm_sequences(df_val, feature_cols, cfg, sequence_length)
    X_test, y_test = create_lstm_sequences(df_test, feature_cols, cfg, sequence_length)

    # Scale features based on training data only
    n_samples, seq_len, n_feat = X_train.shape
    scaler = StandardScaler()
    X_train_2d = X_train.reshape(-1, n_feat)
    scaler.fit(X_train_2d)

    def transform(X):
        X2 = scaler.transform(X.reshape(-1, n_feat))
        return X2.reshape(X.shape)

    X_train_s = transform(X_train)
    X_val_s = transform(X_val)
    X_test_s = transform(X_test)

    model = build_lstm_model((seq_len, n_feat))
    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    model.fit(
        X_train_s,
        y_train,
        validation_data=(X_val_s, y_val),
        epochs=100,
        batch_size=64,
        callbacks=[es],
        verbose=0,
    )

    test_pred = model.predict(X_test_s, verbose=0).reshape(-1)
    metrics = regression_metrics(y_test, test_pred)
    return model, scaler, metrics


# -----------------------------
# Graph utilities (pure PyTorch)
# -----------------------------

def build_knn_adjacency(coords: np.ndarray, k: int = 5) -> np.ndarray:
    """
    Build a symmetric KNN adjacency matrix from [lat, lon].
    Returns dense numpy adjacency with self-loops.
    """
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape [n_nodes, 2].")
    n = coords.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 nodes for a graph.")

    k = max(1, min(k, n - 1))
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nbrs.fit(coords)
    _, indices = nbrs.kneighbors(coords)

    A = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in indices[i, 1:]:
            A[i, j] = 1.0
            A[j, i] = 1.0
    np.fill_diagonal(A, 1.0)
    return A


def normalize_adjacency(A: np.ndarray) -> np.ndarray:
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.power(D, -0.5, where=D > 0)
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.0
    D_mat = np.diag(D_inv_sqrt)
    return D_mat @ A @ D_mat


def check_synchronized_snapshots(df: pd.DataFrame, cfg: Config, min_coverage: float = 0.85) -> bool:
    """
    Returns True if most timestamps have most locations observed.
    This is a practical requirement for graph snapshot models.
    """
    if cfg.location_col not in df.columns:
        return False
    total_locations = df[cfg.location_col].nunique(dropna=True)
    if total_locations < 2:
        return False
    counts = df.groupby("_timestamp")[cfg.location_col].nunique()
    if counts.empty:
        return False
    coverage = counts / total_locations
    return float((coverage >= min_coverage).mean()) >= 0.5


def pivot_snapshot_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    cfg: Config,
) -> Tuple[List[pd.Timestamp], np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Create tensors for graph snapshots:
    X[t, node, feat], y[t, node]
    Only works if each timestamp corresponds to a near-complete set of nodes.
    """
    if cfg.location_col not in df.columns:
        raise ValueError("Graph models require location_name column.")
    if cfg.lat_col not in df.columns or cfg.lon_col not in df.columns:
        raise ValueError("Graph models require latitude and longitude columns.")

    locs = sorted(df[cfg.location_col].dropna().unique().tolist())
    ts_list = sorted(df["_timestamp"].dropna().unique().tolist())

    loc_to_idx = {loc: i for i, loc in enumerate(locs)}
    feat_cols = list(feature_cols)

    X_snapshots = []
    y_snapshots = []
    valid_ts = []

    for ts in ts_list:
        snap = df[df["_timestamp"] == ts].copy()
        # Require reasonably complete coverage
        if snap[cfg.location_col].nunique() < max(2, int(0.8 * len(locs))):
            continue

        snap = snap.set_index(cfg.location_col)
        X_t = np.full((len(locs), len(feat_cols)), np.nan, dtype=np.float32)
        y_t = np.full((len(locs),), np.nan, dtype=np.float32)

        for loc, row in snap.iterrows():
            i = loc_to_idx.get(loc)
            if i is None:
                continue
            vals = pd.to_numeric(row[feat_cols], errors="coerce").to_numpy(dtype=np.float32)
            X_t[i, :] = vals
            y_t[i] = pd.to_numeric(row[cfg.target_col], errors="coerce")

        # Keep only snapshots with enough valid entries
        if np.isfinite(X_t).mean() < 0.85 or np.isfinite(y_t).mean() < 0.85:
            continue

        X_snapshots.append(X_t)
        y_snapshots.append(y_t)
        valid_ts.append(ts)

    if len(X_snapshots) < 5:
        raise ValueError("Not enough synchronized snapshots for graph modeling.")

    X = np.stack(X_snapshots)  # [T, N, F]
    y = np.stack(y_snapshots)  # [T, N]
    return valid_ts, X, y, locs, feat_cols


# -----------------------------
# Pure PyTorch graph models
# -----------------------------

def train_graph_models_if_possible(df: pd.DataFrame, feature_cols: List[str], cfg: Config):
    """
    Optional graph + spatio-temporal models.
    Trains only if the dataset is sufficiently synchronized.
    """
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError:
        print("Torch is not installed; skipping graph models.")
        return None

    if not check_synchronized_snapshots(df, cfg):
        print("Dataset is not sufficiently synchronized for graph models; skipping GNN/ST-GCN.")
        return None

    valid_ts, X, y, locs, feat_cols = pivot_snapshot_features(df, feature_cols, cfg)

    # Split by time snapshots
    nT = X.shape[0]
    tr_end = max(1, int(0.7 * nT))
    va_end = max(tr_end + 1, int(0.85 * nT))
    X_train, y_train = X[:tr_end], y[:tr_end]
    X_val, y_val = X[tr_end:va_end], y[tr_end:va_end]
    X_test, y_test = X[va_end:], y[va_end:]

    if len(X_test) == 0 or len(X_val) == 0:
        print("Not enough snapshots for train/val/test in graph models; skipping.")
        return None

    # Impute with training mean
    feat_mean = np.nanmean(X_train, axis=(0, 1), keepdims=True)
    feat_mean = np.where(np.isnan(feat_mean), 0.0, feat_mean)
    X_train = np.where(np.isnan(X_train), feat_mean, X_train)
    X_val = np.where(np.isnan(X_val), feat_mean, X_val)
    X_test = np.where(np.isnan(X_test), feat_mean, X_test)

    # Scale features
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, X_train.shape[-1]))
    X_train = scaler.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    # Node coordinates from the first available snapshot
    loc_df = (
        df[[cfg.location_col, cfg.lat_col, cfg.lon_col]]
        .dropna()
        .drop_duplicates(subset=[cfg.location_col])
        .set_index(cfg.location_col)
        .loc[locs]
    )
    coords = loc_df[[cfg.lat_col, cfg.lon_col]].to_numpy(dtype=np.float32)
    A = build_knn_adjacency(coords, k=cfg.knn_k)
    A_norm = normalize_adjacency(A)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    A_t = torch.tensor(A_norm, dtype=torch.float32, device=device)
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32, device=device)

    class GCNLayer(nn.Module):
        def __init__(self, in_feats: int, out_feats: int):
            super().__init__()
            self.linear = nn.Linear(in_feats, out_feats)

        def forward(self, X, A):
            # X: [B, N, F]
            AX = torch.matmul(A, X)
            return self.linear(AX)

    class GraphRegressor(nn.Module):
        def __init__(self, in_feats: int):
            super().__init__()
            self.g1 = GCNLayer(in_feats, 64)
            self.g2 = GCNLayer(64, 32)
            self.out = nn.Linear(32, 1)

        def forward(self, X, A):
            h = torch.relu(self.g1(X, A))
            h = torch.relu(self.g2(h, A))
            yhat = self.out(h).squeeze(-1)
            return yhat

    class STGCN(nn.Module):
        def __init__(self, in_feats: int, hidden: int = 32):
            super().__init__()
            self.lstm = nn.LSTM(in_feats, hidden, batch_first=True)
            self.g1 = GCNLayer(hidden, 64)
            self.g2 = GCNLayer(64, 32)
            self.out = nn.Linear(32, 1)

        def forward(self, X, A):
            # X: [B, N, F]  (here we use each snapshot as a batch item after a node-wise summary)
            # For a simple and stable implementation, we first treat node features as a sequence over features.
            # This is a pragmatic spatio-temporal approximation.
            B, N, F = X.shape
            seq = X.reshape(B * N, 1, F)
            h, _ = self.lstm(seq)
            h = h[:, -1, :].reshape(B, N, -1)
            h = torch.relu(self.g1(h, A))
            h = torch.relu(self.g2(h, A))
            yhat = self.out(h).squeeze(-1)
            return yhat

    def train_model(model, Xtr, ytr, Xva, yva, epochs=40, lr=1e-3):
        model = model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        best_val = float("inf")
        best_state = None
        patience = 8
        bad = 0

        for ep in range(epochs):
            model.train()
            opt.zero_grad()
            pred = model(Xtr, A_t)
            loss = loss_fn(pred, ytr)
            loss.backward()
            opt.step()

            model.eval()
            with torch.no_grad():
                vpred = model(Xva, A_t)
                vloss = loss_fn(vpred, yva).item()

            if vloss < best_val - 1e-6:
                best_val = vloss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        return model

    # Train graph-only and ST models
    g_model = GraphRegressor(X_train.shape[-1])
    g_model = train_model(g_model, X_train_t, y_train_t, X_val_t, y_val_t)

    st_model = STGCN(X_train.shape[-1])
    st_model = train_model(st_model, X_train_t, y_train_t, X_val_t, y_val_t)

    def evaluate(model, Xte, yte):
        model.eval()
        with torch.no_grad():
            pred = model(Xte, A_t).detach().cpu().numpy().reshape(-1)
        true = yte.detach().cpu().numpy().reshape(-1)
        return regression_metrics(true, pred)

    g_metrics = evaluate(g_model, X_test_t, y_test_t)
    st_metrics = evaluate(st_model, X_test_t, y_test_t)

    return {
        "graph_regressor": g_metrics,
        "stgcn": st_metrics,
    }


# -----------------------------
# SHAP explainability
# -----------------------------

def run_shap_if_available(model: Pipeline, X_sample: pd.DataFrame):
    """
    Tries SHAP for tree models. If shap isn't installed, simply skips.
    """
    try:
        import shap
    except ImportError:
        print("shap is not installed; skipping SHAP analysis.")
        return None

    # Extract transformed matrix and model
    preprocessor = model.named_steps["preprocessor"]
    tree_model = model.named_steps["model"]

    Xt = preprocessor.transform(X_sample)
    try:
        explainer = shap.Explainer(tree_model, Xt)
        shap_values = explainer(Xt)
        return shap_values
    except Exception:
        # For some xgboost versions, TreeExplainer is more reliable
        try:
            explainer = shap.TreeExplainer(tree_model)
            shap_values = explainer.shap_values(Xt)
            return shap_values
        except Exception as e:
            print(f"SHAP failed: {e}")
            return None


# -----------------------------
# Main execution
# -----------------------------

def main():
    ensure_output_dir(CFG.output_dir)
    print("Loading dataset...")
    df = load_data(CFG)
    print(f"Rows: {len(df):,}, Columns: {df.shape[1]}")

    print("Cleaning and engineering features...")
    df = basic_cleaning(df, CFG)

    numeric_features, categorical_features, spatial_features = make_feature_lists(df, CFG)

    # Build a conservative feature set for XGBoost/LSTM
    # Remove the target and any obviously leaked raw duplicate units if present
    drop_cols = {
        CFG.target_col,
        "temperature_fahrenheit",
        "wind_mph",
        "pressure_in",
        "precip_in",
        "visibility_miles",
        "gust_mph",
        "country",  # constant column
        "last_updated_epoch",
        "last_updated",
        "_timestamp",
    }

    xgb_features = [c for c in (numeric_features + categorical_features) if c not in drop_cols]
    # LSTM uses a fully numeric feature set, so one-hot encode categoricals with pandas
    # For simplicity and robustness, we only use numeric-like columns there.
    lstm_features = [c for c in numeric_features if c not in [CFG.target_col]]

    # Add selected encoded categorical proxies for LSTM
    for cat in ["region", "timezone", "condition_text", "wind_direction", "moon_phase"]:
        if cat in df.columns:
            # Use stable factor codes, with NaN-safe handling
            df[f"{cat}_code"] = df[cat].astype("category").cat.codes.replace(-1, np.nan)
            lstm_features.append(f"{cat}_code")

    # Remove any duplicates while preserving order
    def dedupe(seq):
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    xgb_features = dedupe([c for c in xgb_features if c in df.columns])
    lstm_features = dedupe([c for c in lstm_features if c in df.columns])

    print(f"XGBoost features: {len(xgb_features)}")
    print(f"LSTM features: {len(lstm_features)}")

    train_df, val_df, test_df = train_val_test_split_by_time(df, CFG)

    # XGBoost
    print("\nTraining XGBoost regressor...")
    xgb_model, xgb_metrics = train_xgb(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        numeric_features=[c for c in xgb_features if c in numeric_features],
        categorical_features=[c for c in xgb_features if c in categorical_features],
        cfg=CFG,
    )
    print("XGBoost regression metrics:")
    for k, v in xgb_metrics.items():
        print(f"  {k}: {v:.4f}")

    # XGBoost classification derived from regression output
    try:
        cls_metrics = xgb_classification_report(
            xgb_model, test_df, [c for c in xgb_features if c in df.columns], CFG
        )
        print("XGBoost derived classification metrics:")
        for k, v in cls_metrics.items():
            print(f"  {k}: {v:.4f}")
    except Exception as e:
        print(f"Classification summary skipped: {e}")

    # SHAP
    print("\nRunning SHAP (if available)...")
    try:
        sample_X, _ = make_xy(test_df, [c for c in xgb_features if c in df.columns], CFG)
        sample_X = sample_X.head(min(200, len(sample_X)))
        shap_values = run_shap_if_available(xgb_model, sample_X)
        if shap_values is not None:
            print("SHAP completed successfully.")
    except Exception as e:
        print(f"SHAP skipped due to: {e}")

    # LSTM
    print("\nTraining LSTM model...")
    try:
        lstm_model, lstm_scaler, lstm_metrics = train_lstm_model(
            train_df, val_df, test_df, lstm_features, CFG, CFG.sequence_length
        )
        print("LSTM regression metrics:")
        for k, v in lstm_metrics.items():
            print(f"  {k}: {v:.4f}")
    except Exception as e:
        print(f"LSTM skipped due to: {e}")

    # Graph models
    print("\nTraining graph-based models if data supports them...")
    try:
        graph_results = train_graph_models_if_possible(df, lstm_features, CFG)
        if graph_results is not None:
            for model_name, metrics in graph_results.items():
                print(f"{model_name.upper()} metrics:")
                for k, v in metrics.items():
                    print(f"  {k}: {v:.4f}")
    except Exception as e:
        print(f"Graph models skipped due to: {e}")

    # Save feature list
    pd.Series(xgb_features, name="xgb_features").to_csv(
        os.path.join(CFG.output_dir, "xgb_features.csv"), index=False
    )
    pd.Series(lstm_features, name="lstm_features").to_csv(
        os.path.join(CFG.output_dir, "lstm_features.csv"), index=False
    )

    print(f"\nDone. Outputs saved to: {CFG.output_dir}")


if __name__ == "__main__":
    main()
