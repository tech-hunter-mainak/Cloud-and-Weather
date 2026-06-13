import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ==============================
# Load Original Dataset
# ==============================
df = pd.read_csv("data/IndianWeatherRepository.csv")

print("Original Shape:", df.shape)

# ==============================
# Remove Unnecessary Columns
# ==============================
columns_to_drop = [
    'temperature_fahrenheit',
    'wind_mph',
    'pressure_in',
    'precip_in',
    'feels_like_fahrenheit',
    'visibility_miles',
    'gust_mph',

    'country',
    'location_name',
    'region',
    'timezone',

    'last_updated',

    'condition_text',

    'sunrise',
    'sunset',
    'moonrise',
    'moonset',
    'moon_phase',
    'moon_illumination'
]

df.drop(
    columns=[c for c in columns_to_drop if c in df.columns],
    inplace=True
)

# ==============================
# Select Final Features
# ==============================
selected_features = [
    'latitude',
    'longitude',
    'last_updated_epoch',

    'temperature_celsius',
    'wind_kph',
    'wind_degree',
    'wind_direction',
    'pressure_mb',
    'precip_mm',
    'humidity',

    'feels_like_celsius',
    'visibility_km',
    'uv_index',
    'gust_kph',

    'air_quality_Carbon_Monoxide',
    'air_quality_Ozone',
    'air_quality_Nitrogen_dioxide',
    'air_quality_Sulphur_dioxide',
    'air_quality_PM2.5',
    'air_quality_PM10',
    'air_quality_us-epa-index',
    'air_quality_gb-defra-index',

    'cloud'
]

df = df[selected_features]

print("After Feature Selection:", df.shape)

# ==============================
# Remove Missing Values
# ==============================
df.dropna(inplace=True)

print("After Removing Missing Values:", df.shape)

# ==============================
# Sort by Time
# ==============================
df.sort_values("last_updated_epoch", inplace=True)
df.reset_index(drop=True, inplace=True)

# ==============================
# Encode Wind Direction
# ==============================
le = LabelEncoder()
df['wind_direction'] = le.fit_transform(df['wind_direction'])

# ==============================
# Features and Target
# ==============================
target = 'cloud'

X = df.drop(columns=[target])
y = df[target]

# ==============================
# Scale Data
# ==============================
X_scaler = MinMaxScaler()
X_scaled = X_scaler.fit_transform(X)

y_scaler = MinMaxScaler()
y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1))

# ==============================
# Create Sequences
# ==============================
def create_sequences(X, y, seq_length):
    X_seq = []
    y_seq = []

    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])

    return np.array(X_seq), np.array(y_seq)

SEQ_LENGTH = 24

X_seq, y_seq = create_sequences(
    X_scaled,
    y_scaled,
    SEQ_LENGTH
)

print("Sequence Shape:", X_seq.shape)

# ==============================
# Train/Test Split
# ==============================
split = int(len(X_seq) * 0.8)

X_train = X_seq[:split]
X_test = X_seq[split:]

y_train = y_seq[:split]
y_test = y_seq[split:]

print("Train:", X_train.shape)
print("Test :", X_test.shape)

# ==============================
# Build LSTM
# ==============================
model = Sequential([
    LSTM(
        64,
        return_sequences=True,
        input_shape=(X_train.shape[1], X_train.shape[2])
    ),
    Dropout(0.2),

    LSTM(32),
    Dropout(0.2),

    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

model.summary()

# ==============================
# Train
# ==============================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.1,
    epochs=50,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

# ==============================
# Predict
# ==============================
y_pred_scaled = model.predict(X_test)

y_pred = y_scaler.inverse_transform(y_pred_scaled)
y_true = y_scaler.inverse_transform(y_test)

# ==============================
# Evaluation
# ==============================
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print("\n===== LSTM Results =====")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")