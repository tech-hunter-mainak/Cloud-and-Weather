#!/usr/bin/env python3
"""
Cloud cover prediction with a Keras-safe tabular Transformer.

What this script does:
- Loads the dataset.
- Uses useful weather / air-quality / astronomical / time features as inputs.
- Uses cloud cover as the target.
- Trains a tabular Transformer regressor.
- Saves the model and preprocessing artifacts.
- Predicts from a partial feature dictionary by filling missing values from training statistics.

Key design goals:
- No raw TensorFlow ops on KerasTensors in the Functional graph.
- Partial-feature prediction support.
- Robust handling of missing values, unseen categorical values, and redundant unit columns.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

@dataclass
class Config:
    csv_path: str = "data/IndianWeatherRepository.csv"
    output_dir: str = "outputs_tabular_transformer"

    target_col: str = "cloud"
    timestamp_epoch_col: str = "last_updated_epoch"
    timestamp_text_col: str = "last_updated"

    random_state: int = 42
    train_ratio: float = 0.70
    val_ratio: float = 0.15

    batch_size: int = 256
    epochs: int = 50
    learning_rate: float = 1e-3

    hidden_dim: int = 64
    num_heads: int = 4
    ff_dim: int = 128
    num_blocks: int = 3
    dropout: float = 0.15

    # Feature choices
    use_location_name: bool = True
    use_country: bool = False  # country is constant in your dataset; usually not informative

    # Drop redundant unit columns
    redundant_drop_columns: Tuple[str, ...] = (
        "temperature_fahrenheit",
        "wind_mph",
        "pressure_in",
        "precip_in",
        "visibility_miles",
        "gust_mph",
    )

    # Convert these time strings to minutes since midnight
    time_string_cols: Tuple[str, ...] = (
        "sunrise",
        "sunset",
        "moonrise",
        "moonset",
    )

    # Potential categorical columns
    categorical_candidates: Tuple[str, ...] = (
        "location_name",
        "region",
        "timezone",
        "condition_text",
        "wind_direction",
        "moon_phase",
        "country",
    )


CFG = Config()


# ---------------------------------------------------------------------
# Reproducibility / utilities
# ---------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}
    yt = y_true[mask]
    yp = y_pred[mask]
    return {
        "MAE": float(mean_absolute_error(yt, yp)),
        "RMSE": float(rmse(yt, yp)),
        "R2": float(r2_score(yt, yp)),
    }


def jsonable(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj


# ---------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------

def load_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Loaded dataset is empty.")
    df.columns = [c.strip() for c in df.columns]
    return df


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
        f"Could not parse timestamp from '{cfg.timestamp_epoch_col}' or '{cfg.timestamp_text_col}'."
    )


def time_to_minutes(value: Any) -> float:
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


def cyclical_encode(series: pd.Series, period: float, prefix: str) -> pd.DataFrame:
    x = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    radians = 2.0 * np.pi * (x % period) / period
    return pd.DataFrame(
        {
            f"{prefix}_sin": np.sin(radians),
            f"{prefix}_cos": np.cos(radians),
        }
    )


def drop_redundant_columns(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()
    drop_cols = [c for c in cfg.redundant_drop_columns if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def add_time_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()
    ts = parse_timestamp(df, cfg)
    df["_timestamp"] = ts
    df["_hour"] = ts.dt.hour.astype(float)
    df["_dayofweek"] = ts.dt.dayofweek.astype(float)
    df["_dayofyear"] = ts.dt.dayofyear.astype(float)
    df["_month"] = ts.dt.month.astype(float)

    df = pd.concat([df, cyclical_encode(df["_hour"], 24.0, "hour")], axis=1)
    df = pd.concat([df, cyclical_encode(df["_dayofweek"], 7.0, "dow")], axis=1)
    df = pd.concat([df, cyclical_encode(df["_dayofyear"], 365.0, "doy")], axis=1)
    df = pd.concat([df, cyclical_encode(df["_month"], 12.0, "month")], axis=1)
    return df


def prepare_dataframe(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    if cfg.target_col not in df.columns:
        raise ValueError(f"Target column '{cfg.target_col}' not found.")

    df = drop_redundant_columns(df, cfg)
    df = add_time_features(df, cfg)

    # Convert sunrise/sunset/moonrise/moonset to numeric minutes.
    for col in cfg.time_string_cols:
        if col in df.columns:
            df[f"{col}_minutes"] = df[col].apply(time_to_minutes)

    # Normalize numeric-looking object columns.
    ignore = {
        cfg.target_col,
        "_timestamp",
        cfg.timestamp_epoch_col,
        cfg.timestamp_text_col,
    } | set(cfg.categorical_candidates) | set(cfg.time_string_cols)

    for col in df.columns:
        if col in ignore:
            continue
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().mean() >= 0.70:
                df[col] = converted

    return df


def select_features(df: pd.DataFrame, cfg: Config) -> Tuple[List[str], List[str]]:
    """
    Returns:
        numeric_features, categorical_features
    """
    categorical_features: List[str] = []

    if cfg.use_location_name and "location_name" in df.columns:
        categorical_features.append("location_name")

    for col in ("region", "timezone", "condition_text", "wind_direction", "moon_phase"):
        if col in df.columns:
            categorical_features.append(col)

    if cfg.use_country and "country" in df.columns:
        categorical_features.append("country")

    # Unique preserve order
    categorical_features = list(dict.fromkeys(categorical_features))

    exclude = {
        cfg.target_col,
        cfg.timestamp_epoch_col,
        cfg.timestamp_text_col,
        "_timestamp",
    } | set(categorical_features)

    numeric_features: List[str] = []
    for col in df.columns:
        if col in exclude:
            continue
        if col in cfg.time_string_cols:
            continue
        if col.endswith("_minutes"):
            numeric_features.append(col)
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_features.append(col)

    # Remove any accidental duplicates while preserving order
    numeric_features = list(dict.fromkeys([c for c in numeric_features if c != cfg.target_col]))
    return numeric_features, categorical_features


def split_chronologically(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unique_ts = np.array(sorted(df["_timestamp"].dropna().unique()))
    if len(unique_ts) < 3:
        raise ValueError("Need at least 3 unique timestamps for train/val/test split.")

    n = len(unique_ts)
    train_end = max(1, int(n * cfg.train_ratio))
    val_end = max(train_end + 1, int(n * (cfg.train_ratio + cfg.val_ratio)))
    val_end = min(val_end, n - 1)

    train_ts = unique_ts[:train_end]
    val_ts = unique_ts[train_end:val_end]
    test_ts = unique_ts[val_end:]

    train_df = df[df["_timestamp"].isin(train_ts)].copy()
    val_df = df[df["_timestamp"].isin(val_ts)].copy()
    test_df = df[df["_timestamp"].isin(test_ts)].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Chronological split produced an empty subset. Check timestamp coverage.")

    return train_df, val_df, test_df


def build_vocab(series: pd.Series) -> Dict[str, int]:
    s = series.fillna("UNK").astype(str).str.strip()
    uniq = sorted(v for v in s.unique().tolist() if v and v.lower() != "nan")
    vocab = {"UNK": 0}
    for i, v in enumerate(uniq, start=1):
        vocab[v] = i
    return vocab


def fit_preprocessing(
    train_df: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
) -> Dict[str, Any]:
    artifacts: Dict[str, Any] = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }

    numeric_stats: Dict[str, Dict[str, float]] = {}
    for col in numeric_features:
        s = pd.to_numeric(train_df[col], errors="coerce")
        median = float(s.median()) if s.notna().any() else 0.0
        mean = float(s.mean()) if s.notna().any() else 0.0
        std = float(s.std(ddof=0)) if s.notna().any() else 1.0
        if not np.isfinite(std) or abs(std) < 1e-12:
            std = 1.0
        numeric_stats[col] = {"median": median, "mean": mean, "std": std}
    artifacts["numeric_stats"] = numeric_stats

    categorical_vocab: Dict[str, Dict[str, int]] = {}
    for col in categorical_features:
        categorical_vocab[col] = build_vocab(train_df[col] if col in train_df.columns else pd.Series([], dtype=object))
    artifacts["categorical_vocab"] = categorical_vocab

    return artifacts


def transform_dataframe(
    df: pd.DataFrame,
    artifacts: Dict[str, Any],
    cfg: Config,
) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray]]:
    """
    Convert dataframe to model input dictionaries.
    Missing features are filled using training statistics.
    """
    df = df.copy()
    numeric_features: List[str] = artifacts["numeric_features"]
    categorical_features: List[str] = artifacts["categorical_features"]

    # Ensure every expected feature exists
    for col in numeric_features:
        if col not in df.columns:
            df[col] = np.nan
    for col in categorical_features:
        if col not in df.columns:
            df[col] = np.nan

    x_dict: Dict[str, np.ndarray] = {}

    for col in numeric_features:
        stats = artifacts["numeric_stats"][col]
        s = pd.to_numeric(df[col], errors="coerce").fillna(stats["median"]).astype(float)
        denom = stats["std"] if abs(stats["std"]) > 1e-12 else 1.0
        s = (s - stats["mean"]) / denom
        x_dict[col] = s.to_numpy(dtype=np.float32).reshape(-1, 1)

    for col in categorical_features:
        vocab = artifacts["categorical_vocab"][col]
        s = df[col].fillna("UNK").astype(str).str.strip()
        idx = s.map(lambda v: vocab.get(v, 0)).astype(np.int32)
        x_dict[col] = idx.to_numpy(dtype=np.int32).reshape(-1, 1)

    y = None
    if cfg.target_col in df.columns:
        y = pd.to_numeric(df[cfg.target_col], errors="coerce").to_numpy(dtype=np.float32)

    return x_dict, y


# ---------------------------------------------------------------------
# Keras-safe custom layers
# ---------------------------------------------------------------------

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=max(1, hidden_dim // num_heads),
            dropout=dropout,
        )
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(ff_dim, activation="relu"),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(hidden_dim),
            ]
        )
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = tf.keras.layers.Dropout(dropout)
        self.drop2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=False):
        attn_out = self.attn(x, x, training=training)
        x = self.norm1(x + self.drop1(attn_out, training=training))
        ff = self.ffn(x, training=training)
        x = self.norm2(x + self.drop2(ff, training=training))
        return x


class AddClassToken(tf.keras.layers.Layer):
    def __init__(self, hidden_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        self.cls = self.add_weight(
            name="cls_token",
            shape=(1, 1, self.hidden_dim),
            initializer="random_normal",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        batch_size = tf.shape(x)[0]
        cls = tf.tile(self.cls, [batch_size, 1, 1])
        return tf.concat([cls, x], axis=1)


class AddTokenPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, max_tokens: int, hidden_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.max_tokens = max_tokens
        self.hidden_dim = hidden_dim
        self.pos_emb = tf.keras.layers.Embedding(input_dim=max_tokens, output_dim=hidden_dim)

    def call(self, x):
        # x: [B, T, H]
        seq_len = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        pos = self.pos_emb(positions)  # [T, H]
        return x + pos[tf.newaxis, :, :]


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

def build_model(artifacts: Dict[str, Any], cfg: Config) -> tf.keras.Model:
    numeric_features = artifacts["numeric_features"]
    categorical_features = artifacts["categorical_features"]
    categorical_vocab = artifacts["categorical_vocab"]

    inputs: List[tf.keras.layers.Input] = []
    token_list: List[tf.Tensor] = []

    # Numeric features -> Dense token embeddings
    for col in numeric_features:
        inp = tf.keras.Input(shape=(1,), name=col, dtype=tf.float32)
        x = tf.keras.layers.Dense(cfg.hidden_dim, activation=None)(inp)
        x = tf.keras.layers.Reshape((1, cfg.hidden_dim))(x)
        inputs.append(inp)
        token_list.append(x)

    # Categorical features -> Embeddings
    for col in categorical_features:
        vocab_size = len(categorical_vocab[col]) + 1  # reserve 0 for UNK
        inp = tf.keras.Input(shape=(1,), name=col, dtype=tf.int32)
        x = tf.keras.layers.Embedding(vocab_size, cfg.hidden_dim)(inp)
        x = tf.keras.layers.Reshape((1, cfg.hidden_dim))(x)
        inputs.append(inp)
        token_list.append(x)

    if not token_list:
        raise ValueError("No input features were selected. Check preprocessing logic.")

    # Concatenate feature tokens into a token sequence
    x = tf.keras.layers.Concatenate(axis=1)(token_list)  # [B, num_tokens, H]

    # Add trainable class token and position embeddings
    x = AddClassToken(cfg.hidden_dim)(x)
    x = AddTokenPositionEmbedding(max_tokens=len(token_list) + 1, hidden_dim=cfg.hidden_dim)(x)

    # Transformer blocks
    for _ in range(cfg.num_blocks):
        x = TransformerBlock(cfg.hidden_dim, cfg.num_heads, cfg.ff_dim, cfg.dropout)(x)

    # CLS token output
    x = tf.keras.layers.Lambda(lambda t: t[:, 0, :], output_shape=(cfg.hidden_dim,))(x)
    x = tf.keras.layers.Dense(cfg.ff_dim, activation="relu")(x)
    x = tf.keras.layers.Dropout(cfg.dropout)(x)
    x = tf.keras.layers.Dense(cfg.ff_dim // 2, activation="relu")(x)
    out = tf.keras.layers.Dense(1, name="cloud_prediction")(x)

    model = tf.keras.Model(inputs=inputs, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss="mse",
        metrics=[tf.keras.metrics.MAE],
    )
    return model


# ---------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------

def dict_to_dataset(x_dict: Dict[str, np.ndarray], y: Optional[np.ndarray], batch_size: int, shuffle: bool = False) -> tf.data.Dataset:
    if y is None:
        ds = tf.data.Dataset.from_tensor_slices(x_dict)
    else:
        ds = tf.data.Dataset.from_tensor_slices((x_dict, y))
    if shuffle and y is not None:
        ds = ds.shuffle(min(10000, len(y)), reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def train_model(df: pd.DataFrame, cfg: Config) -> Tuple[tf.keras.Model, Dict[str, Any], Dict[str, float], pd.DataFrame, pd.DataFrame]:
    df = prepare_dataframe(df, cfg)

    # Ensure target exists and is numeric
    df[cfg.target_col] = pd.to_numeric(df[cfg.target_col], errors="coerce")
    df = df[df[cfg.target_col].notna()].copy()

    numeric_features, categorical_features = select_features(df, cfg)
    if cfg.target_col in numeric_features:
        numeric_features.remove(cfg.target_col)
    if cfg.target_col in categorical_features:
        categorical_features.remove(cfg.target_col)

    train_df, val_df, test_df = split_chronologically(df, cfg)
    artifacts = fit_preprocessing(train_df, numeric_features, categorical_features)

    x_train, y_train = transform_dataframe(train_df, artifacts, cfg)
    x_val, y_val = transform_dataframe(val_df, artifacts, cfg)
    x_test, y_test = transform_dataframe(test_df, artifacts, cfg)

    # y arrays
    y_train = y_train.astype(np.float32).reshape(-1, 1)
    y_val = y_val.astype(np.float32).reshape(-1, 1)
    y_test = y_test.astype(np.float32).reshape(-1, 1)

    # Drop any accidental non-finite targets
    train_mask = np.isfinite(y_train[:, 0])
    val_mask = np.isfinite(y_val[:, 0])
    test_mask = np.isfinite(y_test[:, 0])

    x_train = {k: v[train_mask] for k, v in x_train.items()}
    y_train = y_train[train_mask]
    x_val = {k: v[val_mask] for k, v in x_val.items()}
    y_val = y_val[val_mask]
    x_test = {k: v[test_mask] for k, v in x_test.items()}
    y_test = y_test[test_mask]

    model = build_model(artifacts, cfg)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            min_delta=1e-4,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=3,
            factor=0.5,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    train_ds = dict_to_dataset(x_train, y_train, cfg.batch_size, shuffle=True)
    val_ds = dict_to_dataset(x_val, y_val, cfg.batch_size, shuffle=False)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    val_pred = model.predict(x_val, verbose=0).reshape(-1)
    test_pred = model.predict(x_test, verbose=0).reshape(-1)

    val_metrics = regression_metrics(y_val.reshape(-1), val_pred)
    test_metrics = regression_metrics(y_test.reshape(-1), test_pred)

    metrics = {f"val_{k}": v for k, v in val_metrics.items()}
    metrics.update({f"test_{k}": v for k, v in test_metrics.items()})

    artifacts["training_features"] = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }

    # Save history
    ensure_dir(cfg.output_dir)
    pd.DataFrame(history.history).to_csv(os.path.join(cfg.output_dir, "training_history.csv"), index=False)

    return model, artifacts, metrics, train_df, test_df


def save_artifacts(model: tf.keras.Model, artifacts: Dict[str, Any], cfg: Config, metrics: Dict[str, float]) -> None:
    ensure_dir(cfg.output_dir)

    model.save(os.path.join(cfg.output_dir, "cloud_transformer.keras"))

    with open(os.path.join(cfg.output_dir, "preprocessing_artifacts.json"), "w", encoding="utf-8") as f:
        json.dump(jsonable(artifacts), f, indent=2)

    with open(os.path.join(cfg.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(jsonable(asdict(cfg)), f, indent=2)

    pd.DataFrame([metrics]).to_csv(os.path.join(cfg.output_dir, "metrics.csv"), index=False)


def load_artifacts(output_dir: str) -> Tuple[tf.keras.Model, Dict[str, Any], Config]:
    with open(os.path.join(output_dir, "preprocessing_artifacts.json"), "r", encoding="utf-8") as f:
        artifacts = json.load(f)
    with open(os.path.join(output_dir, "config.json"), "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    cfg = Config(**cfg_dict)
    model = tf.keras.models.load_model(
        os.path.join(output_dir, "cloud_transformer.keras"),
        custom_objects={
            "TransformerBlock": TransformerBlock,
            "AddClassToken": AddClassToken,
            "AddTokenPositionEmbedding": AddTokenPositionEmbedding,
        },
    )
    return model, artifacts, cfg


# ---------------------------------------------------------------------
# Inference from partial feature dictionaries
# ---------------------------------------------------------------------

def predict_from_features(
    feature_dict: Dict[str, Any],
    model: tf.keras.Model,
    artifacts: Dict[str, Any],
    cfg: Config,
) -> float:
    """
    Predict cloud cover from a partial or full feature dictionary.

    Rules:
    - Unknown keys are ignored.
    - Missing numeric features are filled with training medians.
    - Missing categorical features are filled with UNK.
    - If a timestamp is provided, engineered time features are generated.
    """
    row = pd.DataFrame([feature_dict]).copy()

    # If a target accidentally appears, ignore it
    if cfg.target_col in row.columns:
        row = row.drop(columns=[cfg.target_col])

    # Generate timestamp-based features when available
    if cfg.timestamp_epoch_col in row.columns or cfg.timestamp_text_col in row.columns:
        try:
            tmp = prepare_dataframe(row.copy(), cfg)
            for col in tmp.columns:
                if col not in row.columns:
                    row[col] = tmp[col]
        except Exception:
            pass

    numeric_features: List[str] = artifacts["numeric_features"]
    categorical_features: List[str] = artifacts["categorical_features"]

    # Add any missing expected features
    for col in numeric_features:
        if col not in row.columns:
            row[col] = np.nan
    for col in categorical_features:
        if col not in row.columns:
            row[col] = np.nan

    x_dict, _ = transform_dataframe(row, artifacts, cfg)
    pred = float(model.predict(x_dict, verbose=0).reshape(-1)[0])
    return pred


def interactive_predict(model: tf.keras.Model, artifacts: Dict[str, Any], cfg: Config) -> None:
    print("\nEnter a JSON dictionary of features.")
    print("Example:")
    print('{"temperature_celsius": 29, "humidity": 78, "pressure_mb": 1008, "latitude": 26.1, "longitude": 91.7}')
    print("Type 'quit' to stop.\n")
    while True:
        raw = input("features> ").strip()
        if raw.lower() in {"quit", "exit"}:
            break
        if not raw:
            continue
        try:
            feature_dict = json.loads(raw)
            pred = predict_from_features(feature_dict, model, artifacts, cfg)
            print(f"Predicted cloud cover: {pred:.2f}")
        except Exception as exc:
            print(f"Could not predict: {exc}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    set_seed(CFG.random_state)
    ensure_dir(CFG.output_dir)

    print("Loading dataset...")
    df = load_data(CFG.csv_path)
    print(f"Rows: {len(df):,}, Columns: {df.shape[1]}")

    print("Training tabular transformer...")
    model, artifacts, metrics, train_df, test_df = train_model(df, CFG)

    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    save_artifacts(model, artifacts, CFG, metrics)
    print(f"\nSaved outputs to: {CFG.output_dir}")

    # Optional interactive mode:
    # interactive_predict(model, artifacts, CFG)


if __name__ == "__main__":
    main()
