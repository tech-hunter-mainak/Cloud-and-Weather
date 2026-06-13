#!/usr/bin/env python3
"""
Tabular Transformer for cloud cover prediction.

What this script does:
- Loads the weather CSV.
- Keeps useful attributes as inputs and cloud cover as the target.
- Drops redundant unit columns.
- Creates time-based features from timestamp fields.
- Trains a tabular transformer regressor.
- Saves model + preprocessing artifacts.
- Predicts from a partial feature dictionary by filling missing values from training statistics.

Missing input features at prediction time are allowed.
Extra unknown features are ignored.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

@dataclass
class CFG:
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

    use_location_name: bool = True
    use_country: bool = False  # country is constant in your dataset, usually not useful

    redundant_drop_columns: Tuple[str, ...] = (
        "temperature_fahrenheit",
        "wind_mph",
        "pressure_in",
        "precip_in",
        "visibility_miles",
        "gust_mph",
    )

    time_string_cols: Tuple[str, ...] = (
        "sunrise",
        "sunset",
        "moonrise",
        "moonset",
    )

    categorical_cols: Tuple[str, ...] = (
        "location_name",
        "region",
        "timezone",
        "condition_text",
        "wind_direction",
        "moon_phase",
        "country",
    )

CFG = CFG()


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metrics_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
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


def parse_timestamp(df: pd.DataFrame, cfg: CFG) -> pd.Series:
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


def cyclical_features(series: pd.Series, period: float, prefix: str) -> pd.DataFrame:
    x = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    radians = 2.0 * np.pi * (x % period) / period
    return pd.DataFrame(
        {
            f"{prefix}_sin": np.sin(radians),
            f"{prefix}_cos": np.cos(radians),
        }
    )


def drop_redundant_columns(df: pd.DataFrame, cfg: CFG) -> pd.DataFrame:
    df = df.copy()
    existing = [c for c in cfg.redundant_drop_columns if c in df.columns]
    if existing:
        df = df.drop(columns=existing)
    return df


# ---------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------

def add_time_features(df: pd.DataFrame, cfg: CFG) -> pd.DataFrame:
    df = df.copy()
    ts = parse_timestamp(df, cfg)
    df["_timestamp"] = ts
    df["_hour"] = ts.dt.hour.astype(float)
    df["_dayofweek"] = ts.dt.dayofweek.astype(float)
    df["_dayofyear"] = ts.dt.dayofyear.astype(float)
    df["_month"] = ts.dt.month.astype(float)

    df = pd.concat([df, cyclical_features(df["_hour"], 24.0, "hour")], axis=1)
    df = pd.concat([df, cyclical_features(df["_dayofweek"], 7.0, "dow")], axis=1)
    df = pd.concat([df, cyclical_features(df["_dayofyear"], 365.0, "doy")], axis=1)
    df = pd.concat([df, cyclical_features(df["_month"], 12.0, "month")], axis=1)
    return df


def prepare_raw_dataframe(df: pd.DataFrame, cfg: CFG) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    if cfg.target_col not in df.columns:
        raise ValueError(f"Target column '{cfg.target_col}' not found.")

    df = drop_redundant_columns(df, cfg)
    df = add_time_features(df, cfg)

    for col in cfg.time_string_cols:
        if col in df.columns:
            df[f"{col}_minutes"] = df[col].apply(time_to_minutes)

    ignore_cols = {
        cfg.target_col,
        "_timestamp",
        cfg.timestamp_epoch_col,
        cfg.timestamp_text_col,
    } | set(cfg.categorical_cols) | set(cfg.time_string_cols)

    for col in df.columns:
        if col in ignore_cols:
            continue
        if df[col].dtype == object:
            maybe_num = pd.to_numeric(df[col], errors="coerce")
            if maybe_num.notna().mean() >= 0.70:
                df[col] = maybe_num

    return df


def infer_feature_columns(df: pd.DataFrame, cfg: CFG) -> Tuple[List[str], List[str]]:
    exclude = {
        cfg.target_col,
        cfg.timestamp_epoch_col,
        cfg.timestamp_text_col,
        "_timestamp",
    }

    categorical = []
    if cfg.use_location_name and "location_name" in df.columns:
        categorical.append("location_name")
    for col in ("region", "timezone", "condition_text", "wind_direction", "moon_phase"):
        if col in df.columns:
            categorical.append(col)
    if cfg.use_country and "country" in df.columns:
        categorical.append("country")
    categorical = list(dict.fromkeys(categorical))

    numeric = []
    for col in df.columns:
        if col in exclude or col in categorical:
            continue
        if col in cfg.time_string_cols:
            continue
        if col.endswith("_minutes"):
            numeric.append(col)
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric.append(col)

    numeric = [c for c in numeric if c in df.columns and c != cfg.target_col]
    return numeric, categorical



def split_by_timestamp(df: pd.DataFrame, cfg: CFG) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ts = np.array(sorted(df["_timestamp"].dropna().unique()))
    if len(ts) < 3:
        raise ValueError("Need at least 3 unique timestamps for train/val/test split.")

    n = len(ts)
    train_n = max(1, int(round(n * cfg.train_ratio)))
    val_n = max(1, int(round(n * cfg.val_ratio)))
    if train_n + val_n >= n:
        # Ensure at least one timestamp remains for test.
        if n >= 4:
            val_n = max(1, n - train_n - 1)
        else:
            train_n = max(1, n - 2)
            val_n = 1

    test_n = n - train_n - val_n
    if test_n < 1:
        test_n = 1
        if train_n > val_n:
            train_n -= 1
        else:
            val_n -= 1

    train_ts = ts[:train_n]
    val_ts = ts[train_n:train_n + val_n]
    test_ts = ts[train_n + val_n:]

    return (
        df[df["_timestamp"].isin(train_ts)].copy(),
        df[df["_timestamp"].isin(val_ts)].copy(),
        df[df["_timestamp"].isin(test_ts)].copy(),
    )

    n = len(ts)
    train_end = int(n * cfg.train_ratio)
    val_end = int(n * (cfg.train_ratio + cfg.val_ratio))

    train_ts = ts[:train_end]
    val_ts = ts[train_end:val_end]
    test_ts = ts[val_end:]

    return (
        df[df["_timestamp"].isin(train_ts)].copy(),
        df[df["_timestamp"].isin(val_ts)].copy(),
        df[df["_timestamp"].isin(test_ts)].copy(),
    )


def build_vocab(series: pd.Series) -> Dict[str, int]:
    vals = series.fillna("UNK").astype(str).str.strip()
    uniq = sorted(set(v for v in vals.tolist() if v and v.lower() != "nan"))
    vocab = {"UNK": 0}
    for i, v in enumerate(uniq, start=1):
        vocab[v] = i
    return vocab


def fit_preprocessing(
    train_df: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
    cfg: CFG,
) -> Dict[str, Any]:
    artifacts: Dict[str, Any] = {}
    artifacts["numeric_features"] = numeric_features
    artifacts["categorical_features"] = categorical_features

    num_stats = {}
    for col in numeric_features:
        s = pd.to_numeric(train_df[col], errors="coerce")
        std = float(s.std(ddof=0)) if s.notna().any() else 1.0
        if not np.isfinite(std) or abs(std) < 1e-12:
            std = 1.0
        num_stats[col] = {
            "median": float(s.median()) if s.notna().any() else 0.0,
            "mean": float(s.mean()) if s.notna().any() else 0.0,
            "std": std,
        }
    artifacts["numeric_stats"] = num_stats

    cat_vocab = {}
    for col in categorical_features:
        cat_vocab[col] = build_vocab(train_df[col] if col in train_df.columns else pd.Series([], dtype=object))
    artifacts["categorical_vocab"] = cat_vocab
    return artifacts


def transform_dataframe(
    df: pd.DataFrame,
    artifacts: Dict[str, Any],
    cfg: CFG,
) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray], pd.DataFrame]:
    df = df.copy()
    numeric_features: List[str] = artifacts["numeric_features"]
    categorical_features: List[str] = artifacts["categorical_features"]

    for col in numeric_features:
        if col not in df.columns:
            df[col] = np.nan
    for col in categorical_features:
        if col not in df.columns:
            df[col] = np.nan

    x_num = {}
    for col in numeric_features:
        stats = artifacts["numeric_stats"][col]
        s = pd.to_numeric(df[col], errors="coerce").fillna(stats["median"]).astype(float)
        s = (s - stats["mean"]) / (stats["std"] if abs(stats["std"]) > 1e-12 else 1.0)
        x_num[col] = s.to_numpy(dtype=np.float32).reshape(-1, 1)

    x_cat = {}
    for col in categorical_features:
        vocab = artifacts["categorical_vocab"][col]
        s = df[col].fillna("UNK").astype(str).str.strip()
        idx = s.map(lambda v: vocab.get(v, 0)).astype(np.int32)
        x_cat[col] = idx.to_numpy(dtype=np.int32).reshape(-1, 1)

    y = None
    if cfg.target_col in df.columns:
        y = pd.to_numeric(df[cfg.target_col], errors="coerce").to_numpy(dtype=np.float32)

    return {**x_num, **x_cat}, y, df


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

class TabularTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(
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
        attn = self.mha(x, x, training=training)
        x = self.norm1(x + self.drop1(attn, training=training))
        ff = self.ffn(x, training=training)
        x = self.norm2(x + self.drop2(ff, training=training))
        return x



def build_model(artifacts: Dict[str, Any], cfg: CFG) -> tf.keras.Model:
    numeric_features = artifacts["numeric_features"]
    categorical_features = artifacts["categorical_features"]
    cat_vocab = artifacts["categorical_vocab"]

    inputs = []
    tokens = []

    feature_bias = tf.keras.layers.Embedding(
        len(numeric_features) + len(categorical_features) + 1, cfg.hidden_dim
    )

    for i, col in enumerate(numeric_features):
        inp = tf.keras.Input(shape=(1,), name=col, dtype=tf.float32)
        x = tf.keras.layers.Dense(cfg.hidden_dim)(inp)
        bias = feature_bias(tf.constant([[i]], dtype=tf.int32))
        bias = tf.squeeze(bias, axis=1)
        x = tf.keras.layers.Add()([x, bias])
        inputs.append(inp)
        tokens.append(x)

    offset = len(numeric_features)
    for j, col in enumerate(categorical_features):
        vocab_size = max(2, len(cat_vocab[col]) + 1)
        inp = tf.keras.Input(shape=(1,), name=col, dtype=tf.int32)
        emb = tf.keras.layers.Embedding(vocab_size, cfg.hidden_dim)(inp)
        emb = tf.keras.layers.Flatten()(emb)
        bias = feature_bias(tf.constant([[offset + j]], dtype=tf.int32))
        bias = tf.squeeze(bias, axis=1)
        x = tf.keras.layers.Add()([emb, bias])
        inputs.append(inp)
        tokens.append(x)

    x = tf.keras.layers.Concatenate(axis=1)([tf.expand_dims(t, axis=1) for t in tokens])

    for _ in range(cfg.num_blocks):
        x = TabularTransformerBlock(cfg.hidden_dim, cfg.num_heads, cfg.ff_dim, cfg.dropout)(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(cfg.ff_dim, activation="relu")(x)
    x = tf.keras.layers.Dropout(cfg.dropout)(x)
    out = tf.keras.layers.Dense(1, name="cloud_prediction")(x)

    model = tf.keras.Model(inputs=inputs, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss="mse",
        metrics=[tf.keras.metrics.MAE],
    )
    return model


# ---------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------

def dict_to_dataset(x_dict: Dict[str, np.ndarray], y: Optional[np.ndarray], batch_size: int, shuffle: bool = False) -> tf.data.Dataset:
    if y is None:
        ds = tf.data.Dataset.from_tensor_slices(x_dict)
    else:
        ds = tf.data.Dataset.from_tensor_slices((x_dict, y))
    if shuffle and y is not None:
        ds = ds.shuffle(min(10000, len(y)), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------
# Training / Evaluation
# ---------------------------------------------------------------------

def train_model(df: pd.DataFrame, cfg: CFG) -> Tuple[tf.keras.Model, Dict[str, Any], Dict[str, float]]:
    df = prepare_raw_dataframe(df, cfg)
    df[cfg.target_col] = pd.to_numeric(df[cfg.target_col], errors="coerce")
    df = df[df[cfg.target_col].notna()].copy()

    numeric_features, categorical_features = infer_feature_columns(df, cfg)
    numeric_features = [c for c in numeric_features if c not in {cfg.target_col, cfg.timestamp_epoch_col, cfg.timestamp_text_col, "_timestamp"}]
    categorical_features = [c for c in categorical_features if c not in {cfg.target_col, cfg.timestamp_epoch_col, cfg.timestamp_text_col, "_timestamp"}]

    train_df, val_df, test_df = split_by_timestamp(df, cfg)
    artifacts = fit_preprocessing(train_df, numeric_features, categorical_features, cfg)

    x_train, y_train, _ = transform_dataframe(train_df, artifacts, cfg)
    x_val, y_val, _ = transform_dataframe(val_df, artifacts, cfg)
    x_test, y_test, _ = transform_dataframe(test_df, artifacts, cfg)

    y_train = y_train.reshape(-1, 1).astype(np.float32)
    y_val = y_val.reshape(-1, 1).astype(np.float32)
    y_test = y_test.reshape(-1, 1).astype(np.float32)

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

    if len(y_val) > 0:
        val_pred = model.predict(x_val, verbose=0).reshape(-1)
        val_metrics = metrics_regression(y_val.reshape(-1), val_pred)
    else:
        val_metrics = {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}

    if len(y_test) > 0:
        test_pred = model.predict(x_test, verbose=0).reshape(-1)
        test_metrics = metrics_regression(y_test.reshape(-1), test_pred)
    else:
        test_pred = np.array([], dtype=np.float32)
        test_metrics = {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}

    metrics = {f"val_{k}": v for k, v in val_metrics.items()}
    metrics.update({f"test_{k}": v for k, v in test_metrics.items()})

    artifacts["feature_schema"] = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }

    pd.DataFrame(history.history).to_csv(os.path.join(cfg.output_dir, "training_history.csv"), index=False)
    return model, artifacts, metrics


def save_artifacts(model: tf.keras.Model, artifacts: Dict[str, Any], cfg: CFG, metrics: Dict[str, float]) -> None:
    ensure_dir(cfg.output_dir)
    model.save(os.path.join(cfg.output_dir, "cloud_transformer.keras"))
    with open(os.path.join(cfg.output_dir, "preprocessing_artifacts.json"), "w", encoding="utf-8") as f:
        json.dump(artifacts, f, indent=2, default=str)
    with open(os.path.join(cfg.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, default=str)
    pd.DataFrame([metrics]).to_csv(os.path.join(cfg.output_dir, "metrics.csv"), index=False)


def load_artifacts(output_dir: str) -> Tuple[tf.keras.Model, Dict[str, Any], CFG]:
    with open(os.path.join(output_dir, "preprocessing_artifacts.json"), "r", encoding="utf-8") as f:
        artifacts = json.load(f)
    with open(os.path.join(output_dir, "config.json"), "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    cfg = CFG(**cfg_dict)
    model = tf.keras.models.load_model(
        os.path.join(output_dir, "cloud_transformer.keras"),
        custom_objects={"TabularTransformerBlock": TabularTransformerBlock},
    )
    return model, artifacts, cfg


# ---------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------

def predict_from_features(
    feature_dict: Dict[str, Any],
    model: tf.keras.Model,
    artifacts: Dict[str, Any],
    cfg: CFG,
) -> float:
    row = pd.DataFrame([feature_dict]).copy()

    if "cloud" in row.columns and cfg.target_col not in row.columns:
        row = row.drop(columns=["cloud"])

    if cfg.timestamp_epoch_col in row.columns or cfg.timestamp_text_col in row.columns:
        tmp = prepare_raw_dataframe(row.copy(), cfg)
        for col in tmp.columns:
            if col not in row.columns:
                row[col] = tmp[col]

    for col in artifacts["numeric_features"]:
        if col not in row.columns:
            row[col] = np.nan
    for col in artifacts["categorical_features"]:
        if col not in row.columns:
            row[col] = np.nan

    x_dict, _, _ = transform_dataframe(row, artifacts, cfg)
    pred = float(model.predict(x_dict, verbose=0).reshape(-1)[0])
    return pred


def interactive_predict(model: tf.keras.Model, artifacts: Dict[str, Any], cfg: CFG) -> None:
    print("\nEnter features as JSON (single row).")
    print("Example:")
    print('{"temperature_celsius": 29, "humidity": 78, "latitude": 26.1, "longitude": 91.7}')
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
    if not os.path.exists(CFG.csv_path):
        raise FileNotFoundError(f"CSV file not found: {CFG.csv_path}")

    df = pd.read_csv(CFG.csv_path)
    print(f"Rows: {len(df):,}, Columns: {df.shape[1]}")

    print("Training tabular transformer...")
    model, artifacts, metrics = train_model(df, CFG)

    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    save_artifacts(model, artifacts, CFG, metrics)
    print(f"\nSaved to: {CFG.output_dir}")

    # Optional interactive mode:
    # interactive_predict(model, artifacts, CFG)


if __name__ == "__main__":
    main()
