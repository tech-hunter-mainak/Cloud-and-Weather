import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from catboost import CatBoostRegressor

# ==========================================
# Load Dataset
# ==========================================
df = pd.read_csv("data/IndianWeatherRepository.csv")

print("Original Shape:", df.shape)

# ==========================================
# Remove Unnecessary Columns
# ==========================================
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
    columns=[col for col in columns_to_drop if col in df.columns],
    inplace=True
)

# ==========================================
# Select Features
# ==========================================
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

# ==========================================
# Remove Missing Values
# ==========================================
df.dropna(inplace=True)

print("After Removing Missing Values:", df.shape)

# ==========================================
# Features and Target
# ==========================================
X = df.drop(columns=['cloud'])
y = df['cloud']

# ==========================================
# Specify Categorical Features
# ==========================================
categorical_features = ['wind_direction']

# Convert categorical columns to string
for col in categorical_features:
    X[col] = X[col].astype(str)

# Get categorical feature indices
cat_features = [
    X.columns.get_loc(col)
    for col in categorical_features
]

# ==========================================
# Train-Test Split
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Train Shape:", X_train.shape)
print("Test Shape :", X_test.shape)

# ==========================================
# Train CatBoost Regressor
# ==========================================
model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=8,
    loss_function='RMSE',
    eval_metric='RMSE',
    random_seed=42,
    verbose=100
)

model.fit(
    X_train,
    y_train,
    cat_features=cat_features,
    eval_set=(X_test, y_test),
    use_best_model=True
)

# ==========================================
# Regression Predictions
# ==========================================
y_pred = model.predict(X_test)

# ==========================================
# Regression Metrics
# ==========================================
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n===== CatBoost Regression Results =====")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# ==========================================
# Classification Metrics (Optional)
# High Cloud Cover = cloud >= 50
# ==========================================
y_test_cls = (y_test >= 50).astype(int)
y_pred_cls = (y_pred >= 50).astype(int)

accuracy = accuracy_score(y_test_cls, y_pred_cls)
precision = precision_score(y_test_cls, y_pred_cls)
recall = recall_score(y_test_cls, y_pred_cls)
f1 = f1_score(y_test_cls, y_pred_cls)

try:
    roc_auc = roc_auc_score(y_test_cls, y_pred)
except:
    roc_auc = np.nan

print("\n===== Classification Metrics =====")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

# ==========================================
# Feature Importance
# ==========================================
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.get_feature_importance()
}).sort_values(by='Importance', ascending=False)

print("\nTop 10 Important Features:")
print(importance.head(10))

# Save Feature Importance
importance.to_csv(
    "catboost_feature_importance.csv",
    index=False
)

print("\nFeature importance saved as:")
print("catboost_feature_importance.csv")