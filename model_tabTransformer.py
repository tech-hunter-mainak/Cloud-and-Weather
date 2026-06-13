import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
    columns=[c for c in columns_to_drop if c in df.columns],
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
df.dropna(inplace=True)

print("Final Shape:", df.shape)

# ==========================================
# Categorical and Numerical Features
# ==========================================
categorical_cols = ['wind_direction']

numerical_cols = [
    c for c in df.columns
    if c not in categorical_cols + ['cloud']
]

# ==========================================
# Encode Categorical Features
# ==========================================
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# ==========================================
# Standardize Numerical Features
# ==========================================
scaler = StandardScaler()

df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# ==========================================
# Prepare Data
# ==========================================
X_cat = df[categorical_cols].values
X_num = df[numerical_cols].values
y = df['cloud'].values

Xcat_train, Xcat_test, Xnum_train, Xnum_test, y_train, y_test = train_test_split(
    X_cat,
    X_num,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# Convert to Tensors
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Xcat_train = torch.LongTensor(Xcat_train).to(device)
Xcat_test = torch.LongTensor(Xcat_test).to(device)

Xnum_train = torch.FloatTensor(Xnum_train).to(device)
Xnum_test = torch.FloatTensor(Xnum_test).to(device)

y_train = torch.FloatTensor(y_train).to(device)
y_test = torch.FloatTensor(y_test).to(device)

# ==========================================
# TabTransformer Model
# ==========================================
class TabTransformer(nn.Module):
    def __init__(
        self,
        categories,
        num_continuous,
        emb_dim=32,
        nhead=4,
        num_layers=2
    ):
        super().__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(cat_size, emb_dim)
            for cat_size in categories
        ])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=nhead,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.mlp = nn.Sequential(
            nn.Linear(
                emb_dim * len(categories) + num_continuous,
                128
            ),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    def forward(self, x_cat, x_num):

        emb = []

        for i, embedding in enumerate(self.embeddings):
            emb.append(embedding(x_cat[:, i]))

        emb = torch.stack(emb, dim=1)

        transformed = self.transformer(emb)

        transformed = transformed.reshape(
            transformed.size(0),
            -1
        )

        x = torch.cat(
            [transformed, x_num],
            dim=1
        )

        return self.mlp(x).squeeze()


# ==========================================
# Initialize Model
# ==========================================
categories = [
    int(df[col].nunique())
    for col in categorical_cols
]

model = TabTransformer(
    categories=categories,
    num_continuous=len(numerical_cols)
).to(device)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# ==========================================
# Training
# ==========================================
epochs = 30
batch_size = 512

for epoch in range(epochs):

    model.train()

    permutation = torch.randperm(
        Xcat_train.size(0),
        device=device
    )

    epoch_loss = 0

    for i in range(0, Xcat_train.size(0), batch_size):

        idx = permutation[i:i+batch_size]

        batch_cat = Xcat_train[idx]
        batch_num = Xnum_train[idx]
        batch_y = y_train[idx]

        optimizer.zero_grad()

        preds = model(
            batch_cat,
            batch_num
        )

        loss = criterion(
            preds,
            batch_y
        )

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{epochs}, "
        f"Loss: {epoch_loss:.4f}"
    )

# ==========================================
# Evaluation
# ==========================================
model.eval()

with torch.no_grad():
    y_pred = model(
        Xcat_test,
        Xnum_test
    ).cpu().numpy()

y_true = y_test.cpu().numpy()

mae = mean_absolute_error(
    y_true,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_true,
        y_pred
    )
)

r2 = r2_score(
    y_true,
    y_pred
)

print("\n===== TabTransformer Results =====")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")