
#!/usr/bin/env python3
"""
Spatio-Temporal Graph Transformer pipeline for cloud cover prediction.

Core idea:
- One graph per timestamp.
- Nodes = locations (lat/lon-based static nodes).
- Edges = K-nearest neighbors by geographic distance with distance/bearing edge features.
- Node features = weather, air quality, astronomical, and time features.
- Temporal windows of consecutive graph snapshots.
- Model = spatial neighborhood transformer + temporal transformer.
- Target = cloud cover at the next timestamp for each node.

The script is designed to be robust and runnable on common Windows Python setups
without relying on PyTorch Geometric or other graph-specific libraries.

It also includes:
- chronological train/val/test split
- masked regression loss to ignore missing labels
- optional XGBoost baseline (only if xgboost is installed)
- safe handling of optional columns
"""

from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    import tensorflow as tf
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "TensorFlow is required for this script. Please install tensorflow."
    ) from exc

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None


# -----------------------------
# Configuration
# -----------------------------

@dataclass
class Config:
    csv_path: str = "data/IndianWeatherRepository.csv"
    output_dir: str = "outputs"

    target_col: str = "cloud"
    timestamp_epoch_col: str = "last_updated_epoch"
    timestamp_text_col: str = "last_updated"

    location_col: str = "location_name"
    lat_col: str = "latitude"
    lon_col: str = "longitude"
    region_col: str = "region"
    timezone_col: str = "timezone"

    lookback: int = 7
    knn_k: int = 10

    train_ratio: float = 0.70
    val_ratio: float = 0.15  # test = remaining 0.15

    batch_size: int = 2
    epochs: int = 30
    learning_rate: float = 1e-3
    random_state: int = 42

    num_spatial_layers: int = 2
    num_temporal_layers: int = 2
    hidden_dim: int = 64
    num_heads: int = 4
    ff_dim: int = 128
    dropout: float = 0.10

    # Columns that are likely categorical and can be one-hot encoded if present
    categorical_cols: Tuple[str, ...] = (
        "condition_text",
        "wind_direction",
        "region",
        "timezone",
        "moon_phase",
    )

    # Time string columns that can be converted to minutes since midnight
    time_string_cols: Tuple[str, ...] = (
        "sunrise",
        "sunset",
        "moonrise",
        "moonset",
    )


CFG = Config()


# -----------------------------
# Small utilities
# -----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def parse_timestamp(df: pd.DataFrame, cfg: Config) -> pd.Series:
    if cfg.timestamp_epoch_col in df.columns:
        ts = pd.to_datetime(df[cfg.timestamp_epoch_col], unit="s", errors="coerce")
        if ts.notna().any():
            return ts
    if cfg.timestamp_text_col in df.columns:
        ts = pd.to_datetime(df[cfg.timestamp_text_col], errors="coerce")
        if ts.notna().any():
            return ts
    raise ValueError(
        f"Could not parse timestamp. Expected '{cfg.timestamp_epoch_col}' or '{cfg.timestamp_text_col}'."
    )


def haversine_distance_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vectorized haversine distance between arrays of coordinates."""
    r = 6371.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def bearing_sin_cos(lat1, lon1, lat2, lon2) -> Tuple[np.ndarray, np.ndarray]:
    """Bearing from point 1 to point 2, returned as sin and cos to handle circularity."""
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.arctan2(y, x)  # [-pi, pi]
    return np.sin(bearing), np.cos(bearing)


def time_to_minutes(value) -> float:
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    if not s:
        return np.nan
    for fmt in ("%H:%M", "%H:%M:%S", "%I:%M %p", "%I:%M:%S %p"):
        try:
            t = pd.to_datetime(s, format=fmt)
            return float(t.hour * 60 + t.minute + t.second / 60.0)
        except Exception:
            pass
    try:
        t = pd.to_datetime(s)
        return float(t.hour * 60 + t.minute + t.second / 60.0)
    except Exception:
        return np.nan


def wrap_angle_to_sin_cos(series: pd.Series, period: float) -> Tuple[pd.Series, pd.Series]:
    radians = 2.0 * np.pi * (series.astype(float) % period) / period
    return np.sin(radians), np.cos(radians)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}
    yt = y_true[mask]
    yp = y_pred[mask]
    return {
        "MAE": float(mean_absolute_error(yt, yp)),
        "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
        "R2": float(r2_score(yt, yp)),
    }


def masked_mse_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """MSE that ignores NaN labels."""
    mask = tf.math.is_finite(y_true)
    y_true_clean = tf.where(mask, y_true, tf.zeros_like(y_true))
    se = tf.square(y_pred - y_true_clean)
    se = tf.where(mask, se, tf.zeros_like(se))
    denom = tf.reduce_sum(tf.cast(mask, tf.float32)) + 1e-6
    return tf.reduce_sum(se) / denom


# -----------------------------
# Data preparation
# -----------------------------

def load_data(cfg: Config) -> pd.DataFrame:
    if not os.path.exists(cfg.csv_path):
        raise FileNotFoundError(f"CSV file not found: {cfg.csv_path}")
    df = pd.read_csv(cfg.csv_path)
    if df.empty:
        raise ValueError("The loaded dataset is empty.")
    df.columns = [c.strip() for c in df.columns]
    return df


def identify_numeric_columns(df: pd.DataFrame, cfg: Config) -> List[str]:
    exclude = {
        cfg.location_col,
        cfg.target_col,
        cfg.timestamp_epoch_col,
        cfg.timestamp_text_col,
    } | set(cfg.categorical_cols) | set(cfg.time_string_cols)

    # Keep obvious numeric columns even if read as object; they are converted later
    numeric_like = []
    for col in df.columns:
        if col in exclude:
            continue
        # location coordinates and weather variables generally live here
        numeric_like.append(col)
    return numeric_like


def convert_time_strings_to_numeric(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()
    for col in cfg.time_string_cols:
        if col in df.columns:
            df[col + "_minutes"] = df[col].apply(time_to_minutes)
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = df["_timestamp"]
    df["_hour"] = ts.dt.hour.astype(float)
    df["_dayofweek"] = ts.dt.dayofweek.astype(float)
    df["_dayofyear"] = ts.dt.dayofyear.astype(float)
    df["_month"] = ts.dt.month.astype(float)

    hour_sin, hour_cos = wrap_angle_to_sin_cos(df["_hour"], 24.0)
    day_sin, day_cos = wrap_angle_to_sin_cos(df["_dayofyear"], 365.0)
    month_sin, month_cos = wrap_angle_to_sin_cos(df["_month"], 12.0)

    df["hour_sin"] = hour_sin
    df["hour_cos"] = hour_cos
    df["day_sin"] = day_sin
    df["day_cos"] = day_cos
    df["month_sin"] = month_sin
    df["month_cos"] = month_cos
    return df


def coerce_numeric_columns(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if col in {cfg.location_col, cfg.target_col, "_timestamp"}:
            continue
        if col in cfg.categorical_cols:
            continue
        if col in cfg.time_string_cols:
            continue
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors="ignore")
            df[col] = converted
    return df


def build_location_table(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    required = [cfg.location_col, cfg.lat_col, cfg.lon_col]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Required spatial column '{col}' not found.")

    loc = (
        df.groupby(cfg.location_col, as_index=False)
        .agg(
            {
                cfg.lat_col: "mean",
                cfg.lon_col: "mean",
                cfg.region_col: (lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]),
                cfg.timezone_col: (lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]),
            }
        )
        if (cfg.region_col in df.columns and cfg.timezone_col in df.columns)
        else df.groupby(cfg.location_col, as_index=False).agg({cfg.lat_col: "mean", cfg.lon_col: "mean"})
    )
    loc = loc.sort_values(cfg.location_col).reset_index(drop=True)
    loc["_node_id"] = np.arange(len(loc), dtype=int)
    return loc


def build_full_panel(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a complete timestamp x location panel.
    Missing observations are introduced as NaNs and then imputed in a time-aware way.
    """
    if cfg.target_col not in df.columns:
        raise ValueError(f"Target column '{cfg.target_col}' not found.")

    df = df.copy()
    df["_timestamp"] = parse_timestamp(df, cfg)
    df = df[df["_timestamp"].notna()].copy()
    df = df.sort_values(["_timestamp", cfg.location_col]).reset_index(drop=True)

    # Convert selected columns to numeric where possible
    df = coerce_numeric_columns(df, cfg)
    df = convert_time_strings_to_numeric(df, cfg)
    df = add_temporal_features(df)

    # Ensure each (timestamp, location) pair appears only once.
    # Some weather feeds contain duplicate rows for the same timestamp/location.
    # We keep the last observation after sorting, which is stable and deterministic.
    df = (
        df.groupby(["_timestamp", cfg.location_col], as_index=False)
        .last()
        .sort_values(["_timestamp", cfg.location_col])
        .reset_index(drop=True)
    )

    # Build location table and keep only consistent coordinates per node
    loc_table = build_location_table(df, cfg)
    loc_order = loc_table[cfg.location_col].tolist()

    unique_ts = np.sort(df["_timestamp"].dropna().unique())
    full_index = pd.MultiIndex.from_product(
        [unique_ts, loc_order], names=["_timestamp", cfg.location_col]
    )

    # Reindex on the unique panel index first; then restore stable location attributes.
    panel = (
        df.set_index(["_timestamp", cfg.location_col])
        .reindex(full_index)
        .reset_index()
        .merge(loc_table, on=cfg.location_col, how="left", suffixes=("", "_loc"))
    )

    # Use the stable location coordinates from the location table
    panel[cfg.lat_col] = panel[cfg.lat_col + "_loc"] if cfg.lat_col + "_loc" in panel.columns else panel[cfg.lat_col]
    panel[cfg.lon_col] = panel[cfg.lon_col + "_loc"] if cfg.lon_col + "_loc" in panel.columns else panel[cfg.lon_col]
    for extra in [cfg.lat_col + "_loc", cfg.lon_col + "_loc"]:
        if extra in panel.columns:
            panel.drop(columns=[extra], inplace=True)

    # Fill location identifiers
    if cfg.region_col in panel.columns and cfg.region_col in loc_table.columns:
        panel = panel.merge(loc_table[[cfg.location_col, cfg.region_col]], on=cfg.location_col, how="left", suffixes=("", "_loc2"))
        if cfg.region_col + "_loc2" in panel.columns:
            panel[cfg.region_col] = panel[cfg.region_col].fillna(panel[cfg.region_col + "_loc2"])
            panel.drop(columns=[cfg.region_col + "_loc2"], inplace=True)

    if cfg.timezone_col in panel.columns and cfg.timezone_col in loc_table.columns:
        panel = panel.merge(loc_table[[cfg.location_col, cfg.timezone_col]], on=cfg.location_col, how="left", suffixes=("", "_loc2"))
        if cfg.timezone_col + "_loc2" in panel.columns:
            panel[cfg.timezone_col] = panel[cfg.timezone_col].fillna(panel[cfg.timezone_col + "_loc2"])
            panel.drop(columns=[cfg.timezone_col + "_loc2"], inplace=True)

    panel = panel.sort_values(["_timestamp", cfg.location_col]).reset_index(drop=True)

    # Ensure temporal features exist after reindex
    panel = add_temporal_features(panel)

    # Preserve the raw target mask before any imputations
    raw_target_mask = panel[cfg.target_col].notna().astype(np.float32).to_numpy()

    # Create a separate historical cloud feature for the model input
    panel["cloud_history"] = pd.to_numeric(panel[cfg.target_col], errors="coerce")

    # Fill categorical columns with a location-specific mode, then global mode
    for col in cfg.categorical_cols:
        if col not in panel.columns:
            continue
        panel[col] = panel.groupby(cfg.location_col)[col].transform(
            lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else np.nan)
        )
        if panel[col].isna().any():
            global_mode = panel[col].mode().iloc[0] if not panel[col].mode().empty else ""
            panel[col] = panel[col].fillna(global_mode)

    # Fill time strings numerically if available
    for col in cfg.time_string_cols:
        num_col = col + "_minutes"
        if num_col in panel.columns:
            panel[num_col] = (
                panel.groupby(cfg.location_col)[num_col]
                .transform(lambda s: s.ffill().bfill())
            )
            if panel[num_col].isna().any():
                panel[num_col] = panel[num_col].fillna(panel[num_col].median())

    # Numeric columns: forward fill over time within location, then fill remaining with location median, then global median
    numeric_cols = []
    for col in panel.columns:
        if col in {
            "_timestamp",
            cfg.location_col,
            cfg.target_col,   # keep the raw target untouched for supervised learning
        } | set(cfg.categorical_cols) | set(cfg.time_string_cols):
            continue
        if pd.api.types.is_numeric_dtype(panel[col]):
            numeric_cols.append(col)

    # Convert object-like numeric columns to numeric
    for col in numeric_cols:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")

    # Time-aware fill per location for all numeric features, including cloud_history as an input feature.
    for col in numeric_cols:
        panel[col] = panel.groupby(cfg.location_col)[col].transform(lambda s: s.ffill().bfill())
        panel[col] = panel.groupby(cfg.location_col)[col].transform(
            lambda s: s.fillna(s.median())
        )
        if panel[col].isna().any():
            panel[col] = panel[col].fillna(panel[col].median())

    # Raw target mask from the original observed target values
    target_mask = raw_target_mask
    panel[cfg.target_col] = pd.to_numeric(panel[cfg.target_col], errors="coerce")

    # Categorical one-hot encoding
    cat_existing = [c for c in cfg.categorical_cols if c in panel.columns]
    panel = pd.get_dummies(panel, columns=cat_existing, dummy_na=False, prefix=cat_existing)

    # Drop non-feature IDs from the panel; they are only used for graph construction
    return panel, loc_table


def build_feature_list(panel: pd.DataFrame, cfg: Config) -> Tuple[List[str], List[str], List[str]]:
    """
    Return feature columns, continuous columns, and dummy columns.

    Notes:
    - The raw target column is excluded from inputs.
    - Historical cloud is represented by the separate `cloud_history` feature.
    - `_node_id` is also excluded because node identity is already represented spatially.
    """
    exclude = {"_timestamp", cfg.location_col, cfg.target_col, "_node_id"}

    feature_cols = [col for col in panel.columns if col not in exclude]

    # Detect one-hot columns by prefix
    dummy_cols = [c for c in feature_cols if any(c.startswith(prefix + "_") for prefix in cfg.categorical_cols)]
    cont_cols = [c for c in feature_cols if c not in dummy_cols]

    return feature_cols, cont_cols, dummy_cols


def build_graph_adjacency(loc_table: pd.DataFrame, cfg: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a KNN graph on the static location coordinates.

    Returns:
        neighbor_idx: [N, K]
        edge_attr: [N, K, 3] => distance_km, sin(bearing), cos(bearing)
        adj_mask: [N, N] boolean adjacency matrix (for diagnostics / optional use)
    """
    if cfg.lat_col not in loc_table.columns or cfg.lon_col not in loc_table.columns:
        raise ValueError("Latitude/longitude columns are required for graph construction.")

    coords = loc_table[[cfg.lat_col, cfg.lon_col]].to_numpy(dtype=np.float64)
    n_nodes = coords.shape[0]
    if n_nodes < 2:
        raise ValueError("Need at least 2 locations to build a graph.")

    k = min(cfg.knn_k, n_nodes - 1)
    # NearestNeighbors with haversine metric expects radians
    coords_rad = np.radians(coords)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="haversine", algorithm="ball_tree")
    nn.fit(coords_rad)
    distances_rad, indices = nn.kneighbors(coords_rad, return_distance=True)

    # First neighbor is the node itself. Remove it.
    neighbor_idx = indices[:, 1:].astype(np.int64)
    dist_km = distances_rad[:, 1:] * 6371.0

    lat1 = coords[:, 0][:, None]
    lon1 = coords[:, 1][:, None]
    lat2 = coords[neighbor_idx][:, :, 0]
    lon2 = coords[neighbor_idx][:, :, 1]
    sin_b, cos_b = bearing_sin_cos(lat1, lon1, lat2, lon2)

    edge_attr = np.stack([dist_km, sin_b, cos_b], axis=-1).astype(np.float32)

    adj = np.zeros((n_nodes, n_nodes), dtype=np.bool_)
    for i in range(n_nodes):
        adj[i, neighbor_idx[i]] = True

    return neighbor_idx, edge_attr, adj


def build_tensors(
    panel: pd.DataFrame,
    loc_table: pd.DataFrame,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[pd.Timestamp], List[str]]:
    """
    Convert panel dataframe into:
    - feature_tensor: [T, N, F]
    - target_tensor:  [T, N]
    - target_mask:    [T, N]
    - timestamps: list of timestamps
    - feature_columns: list of feature names
    """
    panel = panel.copy()
    panel = panel.sort_values(["_timestamp", cfg.location_col]).reset_index(drop=True)

    # Build ordered node list
    node_order = loc_table[cfg.location_col].tolist()
    ts_list = list(pd.Index(panel["_timestamp"].drop_duplicates()).sort_values())
    num_times = len(ts_list)
    num_nodes = len(node_order)

    # After get_dummies, identify feature columns
    feature_cols, cont_cols, dummy_cols = build_feature_list(panel, cfg)

    # Scale only continuous columns
    scaler = StandardScaler()
    cont_cols_existing = [c for c in cont_cols if c in panel.columns and pd.api.types.is_numeric_dtype(panel[c])]
    if cont_cols_existing:
        # Fit on all available rows; this is simple and stable for a research prototype
        panel[cont_cols_existing] = scaler.fit_transform(panel[cont_cols_existing].astype(np.float32))

    # Assemble tensors
    # Re-sort to ensure consistent reshape
    panel = panel.set_index(["_timestamp", cfg.location_col])
    panel = panel.sort_index()

    # Make sure each time has every node in the right order
    ordered_rows = []
    for ts in ts_list:
        block = panel.loc[ts].reindex(node_order)
        block["_timestamp"] = ts
        block[cfg.location_col] = node_order
        ordered_rows.append(block.reset_index(drop=True))
    panel_ordered = pd.concat(ordered_rows, axis=0, ignore_index=True)

    # Final feature columns after ordering
    feature_cols = [
        c for c in panel_ordered.columns
        if c not in {"_timestamp", cfg.location_col, cfg.target_col, "_node_id"}
    ]

    # Numeric conversion safety
    for col in feature_cols:
        if pd.api.types.is_object_dtype(panel_ordered[col]):
            panel_ordered[col] = pd.to_numeric(panel_ordered[col], errors="coerce")
        panel_ordered[col] = panel_ordered[col].astype(np.float32)

    # Fill any residual feature NaNs with 0 (after prior imputation)
    panel_ordered[feature_cols] = panel_ordered[feature_cols].fillna(0.0)

    feature_tensor = panel_ordered[feature_cols].to_numpy(dtype=np.float32).reshape(num_times, num_nodes, -1)
    target_tensor = panel_ordered[cfg.target_col].to_numpy(dtype=np.float32).reshape(num_times, num_nodes)
    target_mask = np.isfinite(target_tensor).astype(np.float32)

    return feature_tensor, target_tensor, target_mask, ts_list, feature_cols


# -----------------------------
# tf.data windowing
# -----------------------------

def make_window_dataset(
    feature_tensor: np.ndarray,
    target_tensor: np.ndarray,
    start_target_idx: int,
    end_target_idx: int,
    lookback: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    """
    Create a dataset of windows:
      X = feature_tensor[t-lookback:t]
      y = target_tensor[t]
    for t in [start_target_idx, end_target_idx).
    """

    indices = np.arange(start_target_idx, end_target_idx, dtype=np.int32)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    num_nodes = feature_tensor.shape[1]
    num_features = feature_tensor.shape[2]

    def generator():
        for t in indices:
            yield feature_tensor[t - lookback : t], target_tensor[t]

    output_signature = (
        tf.TensorSpec(shape=(lookback, num_nodes, num_features), dtype=tf.float32),
        tf.TensorSpec(shape=(num_nodes,), dtype=tf.float32),
    )
    ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(indices), 512), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds


# -----------------------------
# Model components
# -----------------------------

class NeighborhoodTransformerBlock(tf.keras.layers.Layer):
    """
    Spatial transformer over a node's local neighborhood.

    Input x: [B, N, F]
    neighbor_idx: [N, K]
    edge_attr: [N, K, E]
    Output: [B, N, D]
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout

        self.proj = tf.keras.layers.Dense(hidden_dim)
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=max(1, hidden_dim // num_heads),
            dropout=dropout,
        )
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(ff_dim, activation="relu"),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(hidden_dim),
            ]
        )

    def call(self, inputs, training=False):
        x, neighbor_idx, edge_attr = inputs
        # x: [B, N, F]
        # neighbor_idx: [N, K]
        # edge_attr: [N, K, E]
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        neighbor_idx = tf.convert_to_tensor(neighbor_idx, dtype=tf.int32)
        edge_attr = tf.convert_to_tensor(edge_attr, dtype=tf.float32)

        batch_size = tf.shape(x)[0]
        num_nodes = tf.shape(x)[1]
        num_neighbors = tf.shape(neighbor_idx)[1]
        edge_dim = tf.shape(edge_attr)[-1]

        neigh_x = tf.gather(x, neighbor_idx, axis=1)  # [B, N, K, F]

        self_token = tf.expand_dims(x, axis=2)  # [B, N, 1, F]
        self_edge = tf.zeros(tf.stack([batch_size, num_nodes, 1, edge_dim]), dtype=tf.float32)
        self_token = tf.concat([self_token, self_edge], axis=-1)

        edge_broadcast = tf.expand_dims(edge_attr, axis=0)  # [1, N, K, E]
        edge_broadcast = tf.tile(edge_broadcast, [batch_size, 1, 1, 1])
        neigh_token = tf.concat([neigh_x, edge_broadcast], axis=-1)  # [B, N, K, F+E]

        tokens = tf.concat([self_token, neigh_token], axis=2)  # [B, N, K+1, F+E]
        tokens = self.proj(tokens)
        tokens = tf.reshape(tokens, [-1, num_neighbors + 1, self.hidden_dim])  # [B*N, K+1, D]

        attn = self.mha(tokens, tokens, training=training)
        x0 = tokens[:, 0, :]
        x0 = self.norm1(x0 + self.dropout1(attn[:, 0, :], training=training))
        ff = self.ffn(x0, training=training)
        x0 = self.norm2(x0 + self.dropout2(ff, training=training))
        x0 = tf.reshape(x0, [batch_size, num_nodes, self.hidden_dim])
        return x0


class TemporalTransformerBlock(tf.keras.layers.Layer):
    """Temporal transformer block over [B, T, D] sequences."""

    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=max(1, hidden_dim // num_heads),
            dropout=dropout,
        )
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(ff_dim, activation="relu"),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(hidden_dim),
            ]
        )

    def call(self, x, training=False):
        # x: [B, T, D]
        attn = self.mha(x, x, training=training)
        x = self.norm1(x + self.dropout1(attn, training=training))
        ff = self.ffn(x, training=training)
        x = self.norm2(x + self.dropout2(ff, training=training))
        return x


class SpatioTemporalGraphTransformer(tf.keras.Model):
    def __init__(
        self,
        lookback: int,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        num_spatial_layers: int,
        num_temporal_layers: int,
        neighbor_idx: np.ndarray,
        edge_attr: np.ndarray,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lookback = lookback
        self.hidden_dim = hidden_dim
        self.num_spatial_layers = num_spatial_layers
        self.num_temporal_layers = num_temporal_layers

        self.neighbor_idx = tf.constant(neighbor_idx, dtype=tf.int32)
        self.edge_attr = tf.constant(edge_attr, dtype=tf.float32)

        self.spatial_in = tf.keras.layers.Dense(hidden_dim)
        self.spatial_blocks = [
            NeighborhoodTransformerBlock(hidden_dim, num_heads, ff_dim, dropout=dropout)
            for _ in range(num_spatial_layers)
        ]
        self.spatial_residual = tf.keras.layers.Dense(hidden_dim)
        self.spatial_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.time_pos_emb = tf.keras.layers.Embedding(input_dim=lookback, output_dim=hidden_dim)
        self.temporal_blocks = [
            TemporalTransformerBlock(hidden_dim, num_heads, ff_dim, dropout=dropout)
            for _ in range(num_temporal_layers)
        ]
        self.temporal_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.head = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(ff_dim, activation="relu"),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(1),
            ]
        )

    def call(self, inputs, training=False):
        # inputs: [B, T, N, F]
        x_seq = tf.convert_to_tensor(inputs, dtype=tf.float32)
        batch_size = tf.shape(x_seq)[0]
        seq_len = tf.shape(x_seq)[1]
        num_nodes = tf.shape(x_seq)[2]

        spatial_outs = []
        for t in range(self.lookback):
            x_t = x_seq[:, t, :, :]  # [B, N, F]
            x_t = self.spatial_in(x_t)
            residual = x_t
            for block in self.spatial_blocks:
                x_t = block((x_t, self.neighbor_idx, self.edge_attr), training=training)
            x_t = self.spatial_norm(x_t + self.spatial_residual(residual))
            spatial_outs.append(x_t)

        z = tf.stack(spatial_outs, axis=1)  # [B, T, N, D]
        z = tf.transpose(z, perm=[0, 2, 1, 3])  # [B, N, T, D]

        positions = tf.range(self.lookback, dtype=tf.int32)
        pos_emb = self.time_pos_emb(positions)  # [T, D]
        z = z + tf.reshape(pos_emb, [1, 1, self.lookback, self.hidden_dim])

        z = tf.reshape(z, [-1, self.lookback, self.hidden_dim])  # [B*N, T, D]
        for block in self.temporal_blocks:
            z = block(z, training=training)
        z = self.temporal_norm(z)

        last = z[:, -1, :]  # [B*N, D]
        pred = self.head(last, training=training)  # [B*N, 1]
        pred = tf.reshape(pred, [batch_size, num_nodes])
        return pred


# -----------------------------
# Training / evaluation
# -----------------------------

def make_model(num_features: int, lookback: int, neighbor_idx: np.ndarray, edge_attr: np.ndarray, cfg: Config) -> tf.keras.Model:
    model = SpatioTemporalGraphTransformer(
        lookback=lookback,
        hidden_dim=cfg.hidden_dim,
        num_heads=cfg.num_heads,
        ff_dim=cfg.ff_dim,
        num_spatial_layers=cfg.num_spatial_layers,
        num_temporal_layers=cfg.num_temporal_layers,
        neighbor_idx=neighbor_idx,
        edge_attr=edge_attr,
        dropout=cfg.dropout,
    )
    # Build the model with a dummy call
    dummy = tf.zeros([1, lookback, neighbor_idx.shape[0], num_features], dtype=tf.float32)
    _ = model(dummy, training=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss=masked_mse_loss,
    )
    return model


def train_st_model(
    feature_tensor: np.ndarray,
    target_tensor: np.ndarray,
    timestamps: List[pd.Timestamp],
    neighbor_idx: np.ndarray,
    edge_attr: np.ndarray,
    cfg: Config,
) -> Tuple[tf.keras.Model, Dict[str, float], np.ndarray, np.ndarray]:
    num_times = feature_tensor.shape[0]
    if num_times <= cfg.lookback + 2:
        raise ValueError("Not enough timestamps to create training windows.")

    total_samples = num_times - cfg.lookback
    train_end = int(total_samples * cfg.train_ratio)
    val_end = int(total_samples * (cfg.train_ratio + cfg.val_ratio))

    train_start_target = cfg.lookback
    val_start_target = cfg.lookback + train_end
    test_start_target = cfg.lookback + val_end
    test_end_target = num_times

    train_ds = make_window_dataset(
        feature_tensor,
        target_tensor,
        start_target_idx=train_start_target,
        end_target_idx=train_start_target + train_end,
        lookback=cfg.lookback,
        batch_size=cfg.batch_size,
        shuffle=True,
        seed=cfg.random_state,
    )
    val_ds = make_window_dataset(
        feature_tensor,
        target_tensor,
        start_target_idx=val_start_target,
        end_target_idx=val_start_target + (val_end - train_end),
        lookback=cfg.lookback,
        batch_size=cfg.batch_size,
        shuffle=False,
        seed=cfg.random_state,
    )
    test_ds = make_window_dataset(
        feature_tensor,
        target_tensor,
        start_target_idx=test_start_target,
        end_target_idx=test_end_target,
        lookback=cfg.lookback,
        batch_size=cfg.batch_size,
        shuffle=False,
        seed=cfg.random_state,
    )

    num_features = feature_tensor.shape[-1]
    model = make_model(num_features, cfg.lookback, neighbor_idx, edge_attr, cfg)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            min_delta=1e-4,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=3,
            factor=0.5,
            min_lr=1e-5,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Predict on validation and test windows
    def collect_predictions(ds: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        y_true_all = []
        y_pred_all = []
        for x_batch, y_batch in ds:
            pred = model(x_batch, training=False).numpy()
            y_true_all.append(y_batch.numpy())
            y_pred_all.append(pred)
        if not y_true_all:
            return np.empty((0, target_tensor.shape[1]), dtype=np.float32), np.empty((0, target_tensor.shape[1]), dtype=np.float32)
        return np.concatenate(y_true_all, axis=0), np.concatenate(y_pred_all, axis=0)

    y_val_true, y_val_pred = collect_predictions(val_ds)
    y_test_true, y_test_pred = collect_predictions(test_ds)

    val_metrics = regression_metrics(y_val_true.reshape(-1), y_val_pred.reshape(-1))
    test_metrics = regression_metrics(y_test_true.reshape(-1), y_test_pred.reshape(-1))

    metrics = {f"val_{k}": v for k, v in val_metrics.items()}
    metrics.update({f"test_{k}": v for k, v in test_metrics.items()})

    # Save training history for inspection
    pd.DataFrame(history.history).to_csv(os.path.join(cfg.output_dir, "training_history.csv"), index=False)

    return model, metrics, y_test_true, y_test_pred


# -----------------------------
# Optional baseline
# -----------------------------

def train_xgboost_baseline(
    feature_tensor: np.ndarray,
    target_tensor: np.ndarray,
    cfg: Config,
) -> Optional[Dict[str, float]]:
    """
    Optional baseline that uses the last observed graph snapshot in each window
    to predict the next timestamp's cloud cover per node.

    This is a simpler comparator than the spatio-temporal model.
    """
    if XGBRegressor is None:
        print("xgboost is not installed; skipping baseline.")
        return None

    num_times = feature_tensor.shape[0]
    if num_times <= cfg.lookback + 2:
        return None

    X_rows = []
    y_rows = []
    ts_idx_rows = []

    for t in range(cfg.lookback, num_times):
        x_last = feature_tensor[t - 1]  # [N, F]
        y = target_tensor[t]  # [N]
        mask = np.isfinite(y)
        X_rows.append(x_last[mask])
        y_rows.append(y[mask])
        ts_idx_rows.append(np.full(mask.sum(), t, dtype=np.int32))

    if not X_rows:
        return None

    X = np.concatenate(X_rows, axis=0)
    y = np.concatenate(y_rows, axis=0)

    # Chronological split by target timestamp index
    ts_idx = np.concatenate(ts_idx_rows, axis=0)
    unique_t = np.unique(ts_idx)
    train_end_t = unique_t[int(len(unique_t) * cfg.train_ratio)]
    val_end_t = unique_t[int(len(unique_t) * (cfg.train_ratio + cfg.val_ratio))]

    train_mask = ts_idx < train_end_t
    val_mask = (ts_idx >= train_end_t) & (ts_idx < val_end_t)
    test_mask = ts_idx >= val_end_t

    model = XGBRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=cfg.random_state,
        n_jobs=max(1, os.cpu_count() or 1),
    )
    model.fit(X[train_mask], y[train_mask])

    val_pred = model.predict(X[val_mask]) if val_mask.any() else np.array([])
    test_pred = model.predict(X[test_mask]) if test_mask.any() else np.array([])

    metrics = {}
    if val_mask.any():
        metrics.update({f"val_{k}": v for k, v in regression_metrics(y[val_mask], val_pred).items()})
    if test_mask.any():
        metrics.update({f"test_{k}": v for k, v in regression_metrics(y[test_mask], test_pred).items()})

    return metrics


# -----------------------------
# Main
# -----------------------------

def main():
    set_global_seed(CFG.random_state)
    ensure_dir(CFG.output_dir)

    print("Loading dataset...")
    df = load_data(CFG)
    print(f"Rows: {len(df):,}, Columns: {df.shape[1]}")

    print("Building full timestamp-location panel...")
    panel, loc_table = build_full_panel(df, CFG)
    print(f"Panel rows: {len(panel):,}")
    print(f"Unique locations: {len(loc_table):,}")

    print("Constructing graph...")
    neighbor_idx, edge_attr, adj = build_graph_adjacency(loc_table, CFG)
    print(f"Graph nodes: {neighbor_idx.shape[0]}, neighbors per node: {neighbor_idx.shape[1]}")

    print("Building tensors...")
    feature_tensor, target_tensor, target_mask, timestamps, feature_cols = build_tensors(panel, loc_table, CFG)
    print(f"Feature tensor shape: {feature_tensor.shape}")
    print(f"Target tensor shape: {target_tensor.shape}")
    print(f"Feature count: {len(feature_cols)}")

    # Save graph metadata
    pd.DataFrame(
        {
            "graph_id": np.arange(len(timestamps), dtype=int),
            "timestamp": pd.to_datetime(timestamps),
        }
    ).to_csv(os.path.join(CFG.output_dir, "graph_timestamps.csv"), index=False)

    # Optional baseline
    print("\nTraining optional XGBoost baseline...")
    try:
        xgb_metrics = train_xgboost_baseline(feature_tensor, target_tensor, CFG)
        if xgb_metrics is not None:
            print("XGBoost baseline metrics:")
            for k, v in xgb_metrics.items():
                print(f"  {k}: {v:.4f}")
            pd.DataFrame([xgb_metrics]).to_csv(os.path.join(CFG.output_dir, "xgboost_baseline_metrics.csv"), index=False)
        else:
            print("XGBoost baseline skipped.")
    except Exception as exc:
        print(f"XGBoost baseline skipped due to: {exc}")

    print("\nTraining spatio-temporal graph transformer...")
    model, metrics, y_test_true, y_test_pred = train_st_model(
        feature_tensor=feature_tensor,
        target_tensor=target_tensor,
        timestamps=timestamps,
        neighbor_idx=neighbor_idx,
        edge_attr=edge_attr,
        cfg=CFG,
    )

    print("Spatio-temporal transformer metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    pd.DataFrame([metrics]).to_csv(os.path.join(CFG.output_dir, "stgt_metrics.csv"), index=False)

    # Save predictions for later analysis
    pred_df = pd.DataFrame(
        {
            "true_cloud": y_test_true.reshape(-1),
            "pred_cloud": y_test_pred.reshape(-1),
        }
    )
    pred_df = pred_df[np.isfinite(pred_df["true_cloud"])]
    pred_df.to_csv(os.path.join(CFG.output_dir, "stgt_test_predictions.csv"), index=False)

    # Save model
    try:
        model.save(os.path.join(CFG.output_dir, "stgt_model.keras"))
    except Exception as exc:
        print(f"Model save skipped due to: {exc}")

    print(f"\nDone. Outputs saved to: {CFG.output_dir}")


if __name__ == "__main__":
    main()
