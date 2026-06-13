#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║  Spatiotemporal Graph Transformer — Cloud Cover Prediction               ║
║  Indian Weather Repository                                               ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Each weather station is a graph node (lat/lon).                        ║
║  Directed edges carry distance + bearing between every pair of           ║
║  K-nearest-neighbour stations.                                           ║
║                                                                          ║
║  One graph snapshot per timestamp.  Graphs are ordered by time.         ║
║                                                                          ║
║  Architecture:                                                           ║
║    Input  (B, W, N, F)  — B batches, W timesteps, N nodes, F features   ║
║    ┌──────────────────────────────────────────────────────────┐          ║
║    │  NodeEmbedding      F → d_model                         │          ║
║    │  GraphTransformerLayer × n_gl  (spatial, sparse KNN)    │          ║
║    │    Edge-aware multi-head attention:                      │          ║
║    │      Q from destination, K/V from source + edge feat     │          ║
║    │      Segment-softmax over each node's incoming edges     │          ║
║    │  TemporalTransformerLayer × n_tl  (per-node, over W)    │          ║
║    │  PredictionHead  d_model → 1                            │          ║
║    └──────────────────────────────────────────────────────────┘          ║
║    Output (B, N)  — cloud cover at T+1 for every node                   ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════════════════
# §0  IMPORTS
# ═══════════════════════════════════════════════════════════════════════════
import os
import math
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import BallTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)


# ═══════════════════════════════════════════════════════════════════════════
# §1  CONFIGURATION  — edit here to tune the run
# ═══════════════════════════════════════════════════════════════════════════
class Cfg:
    # ── paths ──────────────────────────────────────────────────────────────
    CSV_PATH = "IndianWeatherRepository.csv"   # path to dataset
    OUT_DIR  = "outputs"                        # where to save model & plots

    # ── graph ──────────────────────────────────────────────────────────────
    MIN_NODES_PER_TS = 400   # only use "dense" collection batches
    CORE_THRESH      = 1.00  # location must appear in 100 % of dense batches
    KNN_K            = 20    # directed edges per node (K nearest neighbours)
    #   → each node i gets K incoming edges from K spatially closest nodes
    #   → edge feature: [normalised_distance, sin(bearing), cos(bearing)]

    # ── temporal ───────────────────────────────────────────────────────────
    WINDOW = 6               # look-back window W (in timestamps)
    #   Input  : graphs at t-W+1 … t
    #   Target : cloud cover at t+1 for every node

    # ── model ──────────────────────────────────────────────────────────────
    D_MODEL = 64             # hidden dimension
    N_HEADS = 4              # attention heads  (D_MODEL must be divisible)
    N_GL    = 2              # number of Graph Transformer layers (spatial)
    N_TL    = 2              # number of Temporal Transformer layers
    DROPOUT = 0.10

    # ── training ───────────────────────────────────────────────────────────
    BATCH   = 4
    EPOCHS  = 50
    LR      = 3e-4
    WD      = 1e-4           # AdamW weight decay
    TRAIN_F = 0.70           # temporal train split
    VAL_F   = 0.15           # temporal val split  (rest → test)
    PATIENCE = 10            # early-stopping patience (val MSE)
    DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"


# ═══════════════════════════════════════════════════════════════════════════
# §2  GEO UTILITIES
# ═══════════════════════════════════════════════════════════════════════════
def haversine_km(la1: float, lo1: float, la2: float, lo2: float) -> float:
    """Great-circle distance in km between two (lat, lon) points."""
    R = 6371.0
    la1, lo1, la2, lo2 = map(math.radians, [la1, lo1, la2, lo2])
    a = (math.sin((la2 - la1) / 2) ** 2
         + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(max(0.0, min(1.0, a))))


def bearing_rad(la1: float, lo1: float, la2: float, lo2: float) -> float:
    """Initial compass bearing in radians from point 1 to point 2."""
    la1, lo1, la2, lo2 = map(math.radians, [la1, lo1, la2, lo2])
    dlo = lo2 - lo1
    x = math.sin(dlo) * math.cos(la2)
    y = math.cos(la1) * math.sin(la2) - math.sin(la1) * math.cos(la2) * math.cos(dlo)
    return math.atan2(x, y)


# ═══════════════════════════════════════════════════════════════════════════
# §3  DATA PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════
def preprocess(path: str) -> pd.DataFrame:
    """
    Load CSV and engineer features.

    Dropped  : duplicate-unit cols (°F, mph, in, miles), high-cardinality ids,
               astronomical event times, AQI index codes.
    Encoded  : wind direction → (sin, cos); condition / moon_phase / region
               → integer label; time → cyclic (sin, cos).
    Transformed: pollution cols → log1p to reduce right skew.
    """
    print("  Loading CSV …")
    df = pd.read_csv(path)
    df["last_updated"] = pd.to_datetime(df["last_updated"])
    df = df.sort_values("last_updated").reset_index(drop=True)

    # ── drop redundant / leakage columns ──────────────────────────────────
    drop_cols = [
        "country", "timezone", "last_updated_epoch", "location_name",
        "temperature_fahrenheit", "wind_mph", "pressure_in",
        "precip_in", "visibility_miles", "gust_mph", "feels_like_fahrenheit",
        "sunrise", "sunset", "moonrise", "moonset",
        "air_quality_us-epa-index", "air_quality_gb-defra-index",
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # ── wind direction: cyclic encoding from numeric degrees ───────────────
    df["wd_sin"] = np.sin(np.radians(df["wind_degree"].fillna(0.0)))
    df["wd_cos"] = np.cos(np.radians(df["wind_degree"].fillna(0.0)))
    df.drop(columns=["wind_direction", "wind_degree"], inplace=True)

    # ── cyclic time features ───────────────────────────────────────────────
    df["hr_sin"]  = np.sin(2 * np.pi * df["last_updated"].dt.hour / 24)
    df["hr_cos"]  = np.cos(2 * np.pi * df["last_updated"].dt.hour / 24)
    df["mo_sin"]  = np.sin(2 * np.pi * df["last_updated"].dt.month / 12)
    df["mo_cos"]  = np.cos(2 * np.pi * df["last_updated"].dt.month / 12)
    df["doy_sin"] = np.sin(2 * np.pi * df["last_updated"].dt.dayofyear / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["last_updated"].dt.dayofyear / 365)

    # ── label-encode categoricals ─────────────────────────────────────────
    for col in ["condition_text", "moon_phase", "region"]:
        le = {v: i for i, v in enumerate(sorted(df[col].dropna().unique()))}
        df[col + "_enc"] = df[col].map(le).fillna(-1.0).astype(np.float32)
        df.drop(columns=[col], inplace=True)

    # ── log-transform skewed pollution features ────────────────────────────
    pollution_cols = [
        "air_quality_Carbon_Monoxide", "air_quality_Ozone",
        "air_quality_Nitrogen_dioxide", "air_quality_Sulphur_dioxide",
        "air_quality_PM2.5", "air_quality_PM10",
    ]
    for col in pollution_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))

    return df


# ═══════════════════════════════════════════════════════════════════════════
# §4  LOCATION SELECTION
# ═══════════════════════════════════════════════════════════════════════════
def get_core_locations(
    df: pd.DataFrame,
    min_nodes: int,
    core_thresh: float,
) -> tuple:
    """
    Identify the "dense" collection timestamps and the core node set.

    Dense timestamp : one where ≥ min_nodes locations were recorded
                      simultaneously (the dataset has ~123 such daily batches).
    Core location   : a (lat, lon) pair present in ≥ core_thresh × 100 %
                      of all dense timestamps.

    Returns
    -------
    core_locs : DataFrame with columns [latitude, longitude, node_id]
    dense_ts  : sorted list of dense timestamps
    """
    rpt      = df.groupby("last_updated").size()
    dense_ts = sorted(rpt[rpt >= min_nodes].index)

    df_dense = df[df["last_updated"].isin(dense_ts)]
    loc_ts   = df_dense.groupby(["latitude", "longitude"])["last_updated"].nunique()

    threshold = core_thresh * len(dense_ts)
    core      = loc_ts[loc_ts >= threshold].reset_index()[["latitude", "longitude"]]
    core      = core.reset_index(drop=True)
    core["node_id"] = core.index.astype(np.int64)

    print(f"  Dense timestamps : {len(dense_ts)}")
    print(f"  Core nodes (N)   : {len(core)}")
    return core, dense_ts


# ═══════════════════════════════════════════════════════════════════════════
# §5  GRAPH CONSTRUCTION  (static — same topology for every timestamp)
# ═══════════════════════════════════════════════════════════════════════════
def build_knn_graph(
    core_locs: pd.DataFrame,
    K: int,
) -> tuple:
    """
    Build a K-nearest-neighbour directed graph over the N weather stations.

    For each destination node j, find its K nearest source nodes i and add
    directed edges i → j.  Edge features:
        [normalised_distance, sin(bearing_i→j), cos(bearing_i→j)]

    Returns
    -------
    edge_index : np.int64  (2, N·K)   row 0 = src, row 1 = dst
    edge_attr  : np.float32 (N·K, 3)
    """
    lats = core_locs["latitude"].values
    lons = core_locs["longitude"].values
    N    = len(core_locs)

    # BallTree on (lat, lon) in radians with haversine metric
    coords_rad = np.radians(np.stack([lats, lons], axis=1))
    tree       = BallTree(coords_rad, metric="haversine")
    dist_rad, nbr_idx = tree.query(coords_rad, k=K + 1)  # k+1: first col is self

    R_KM = 6371.0
    src_list, dst_list, feat_list = [], [], []

    for j in range(N):                        # j = destination (aggregator)
        for ki in range(1, K + 1):            # skip k=0 (self)
            i   = int(nbr_idx[j, ki])         # i = source (provider)
            d   = dist_rad[j, ki] * R_KM      # haversine distance in km
            b   = bearing_rad(lats[i], lons[i], lats[j], lons[j])  # i→j bearing
            src_list.append(i)
            dst_list.append(j)
            feat_list.append([d, math.sin(b), math.cos(b)])

    edge_index = np.array([src_list, dst_list], dtype=np.int64)    # (2, M)
    edge_attr  = np.array(feat_list, dtype=np.float32)              # (M, 3)

    # Normalise distance to [0, 1]
    max_d = edge_attr[:, 0].max()
    edge_attr[:, 0] /= max_d + 1e-9

    print(f"  Edges  (M = N×K) : {edge_index.shape[1]}  "
          f"({N} nodes × {K} neighbours)")
    return edge_index, edge_attr


# ═══════════════════════════════════════════════════════════════════════════
# §6  NODE-FEATURE TENSORS
# ═══════════════════════════════════════════════════════════════════════════
def build_tensors(
    df: pd.DataFrame,
    core_locs: pd.DataFrame,
    dense_ts: list,
    target: str = "cloud",
) -> tuple:
    """
    Construct node-feature tensor X_all (T, N, F) and target y_all (T, N).

    • Only dense timestamps where ALL N core nodes are present are kept.
    • `target` (cloud cover) is included as a feature so the model sees
      past cloud cover when predicting future cloud cover.
    • Raw (un-scaled) arrays are returned; normalisation is done after
      the temporal train/val/test split (to prevent leakage).

    Returns
    -------
    X_all     : float32 (T, N, F)
    y_all     : float32 (T, N)
    feat_cols : list[str]  — feature column names (F columns)
    full_ts   : list       — timestamps retained
    """
    N = len(core_locs)

    # ── keep only core locations ───────────────────────────────────────────
    df = df.merge(
        core_locs[["latitude", "longitude", "node_id"]],
        on=["latitude", "longitude"],
        how="inner",
    )
    df = df[df["last_updated"].isin(dense_ts)].copy()

    # ── deduplicate: average if a location appears twice in one timestamp ──
    id_cols  = ["last_updated", "node_id"]
    num_cols = [c for c in df.columns
                if c not in id_cols + ["latitude", "longitude"]
                and pd.api.types.is_numeric_dtype(df[c])]
    df = df.groupby(id_cols)[num_cols].mean().reset_index()

    # ── keep only timestamps where every core node is present ──────────────
    ts_counts = df.groupby("last_updated")["node_id"].nunique()
    full_ts   = sorted(ts_counts[ts_counts == N].index)
    df        = df[df["last_updated"].isin(full_ts)].copy()
    T         = len(full_ts)
    print(f"  Full timestamps  : {T}  (all {N} nodes present)")

    # ── feature columns: every numeric column except identifiers ────────────
    feat_cols = [c for c in num_cols if c not in ["node_id"]]
    # ensure `target` is in features (past cloud helps predict future cloud)
    if target not in feat_cols:
        feat_cols.append(target)

    # ── reshape to (T, N, F) using sorted order ────────────────────────────
    df = df.sort_values(["last_updated", "node_id"]).reset_index(drop=True)
    assert len(df) == T * N, (
        f"Expected {T * N} rows after sort; got {len(df)}. "
        "Check for duplicate (timestamp, node_id) entries."
    )

    F_dim  = len(feat_cols)
    X_all  = df[feat_cols].values.reshape(T, N, F_dim).astype(np.float32)
    y_all  = df[target].values.reshape(T, N).astype(np.float32)

    print(f"  X_all shape      : {X_all.shape}   (T × N × F)")
    print(f"  Features ({F_dim})    : {feat_cols}")
    return X_all, y_all, feat_cols, full_ts


# ═══════════════════════════════════════════════════════════════════════════
# §7  PYTORCH DATASET
# ═══════════════════════════════════════════════════════════════════════════
class WeatherGraphDataset(Dataset):
    """
    Sliding-window dataset over a temporal sequence of graph snapshots.

    Item i  →  (X_seq, y_next)
      X_seq  : (W, N, F)  past W snapshots of all node features
      y_next : (N,)       cloud cover at the W+1-th snapshot  ← prediction target
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, window: int):
        self.X = torch.from_numpy(X)   # (T, N, F)
        self.y = torch.from_numpy(y)   # (T, N)
        self.W = window

    def __len__(self) -> int:
        return len(self.X) - self.W

    def __getitem__(self, idx: int):
        return self.X[idx: idx + self.W], self.y[idx + self.W]


# ═══════════════════════════════════════════════════════════════════════════
# §8  MODEL
# ═══════════════════════════════════════════════════════════════════════════

# ── §8a  Sparse Graph Transformer Layer ──────────────────────────────────────
class GraphTransformerLayer(nn.Module):
    """
    Edge-aware multi-head sparse graph attention.

    For directed edge e : src(i) → dst(j):
        Q_j   = W_Q · h_j                      (destination query)
        K_e   = W_K · h_i  +  W_Ke · e_attr   (source key  + edge bias)
        V_e   = W_V · h_i  +  W_Ve · e_attr   (source value + edge bias)
        score = (Q_j · K_e) / √d_k
        α_e   = softmax_{ e ∈ in-edges(j) }(score)   [segment softmax]
        h_j'  = Σ_{ e ∈ in-edges(j) } α_e · V_e

    This formulation lets the model learn:
      • Which nearby stations are most informative for predicting cloud cover
        at the destination (via attention weights α).
      • How direction and distance modulate that relevance (via W_Ke, W_Ve).

    Memory: O(B · M · H · d_k)  with M = N·K edges  — scales linearly with N.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        edge_dim: int,
        d_ff: int,
        dropout: float,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d  = d_model
        self.H  = n_heads
        self.dk = d_model // n_heads

        # Node projections
        self.W_Q  = nn.Linear(d_model, d_model, bias=False)
        self.W_K  = nn.Linear(d_model, d_model, bias=False)
        self.W_V  = nn.Linear(d_model, d_model, bias=False)
        self.W_O  = nn.Linear(d_model, d_model)

        # Edge projections  (edge_dim → d_model, split into heads internally)
        self.W_Ke = nn.Linear(edge_dim, d_model, bias=False)
        self.W_Ve = nn.Linear(edge_dim, d_model, bias=False)

        # Feed-forward sub-layer
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    # ── segment softmax (pure PyTorch, no external scatter libs) ─────────────
    @staticmethod
    def _segment_softmax(
        scores: torch.Tensor,   # (B, M, H) — clamped to [-10, 10]
        dst_idx: torch.Tensor,  # (M,)       — destination node of each edge
        N: int,
    ) -> torch.Tensor:          # (B, M, H) — normalised attention weights
        """
        Softmax within each destination-node segment.

        Because scores are pre-clamped to [-10, 10], exp() is bounded to
        [4.5e-5, 2.2e4] — no overflow, and the result is numerically stable
        without a per-segment max subtraction.

        Implementation: two scatter_add_ calls (sum of exp, then normalise).
        scatter_add_ is available in all PyTorch versions ≥ 1.0.
        """
        B, M, H = scores.shape
        exp_s    = scores.exp()                           # (B, M, H)

        seg_sum  = exp_s.new_zeros(B, N, H)
        idx_exp  = dst_idx.view(1, M, 1).expand(B, M, H)
        seg_sum.scatter_add_(1, idx_exp, exp_s)           # sum per dst node

        # normalise: divide each edge's exp by its destination's sum
        return exp_s / (seg_sum[:, dst_idx, :] + 1e-9)   # (B, M, H)

    def forward(
        self,
        H: torch.Tensor,            # (B, N, d_model)
        edge_index: torch.Tensor,   # (2, M)  long
        edge_attr: torch.Tensor,    # (M, 3)  float
    ) -> torch.Tensor:              # (B, N, d_model)

        B, N, _ = H.shape
        src = edge_index[0]         # (M,)
        dst = edge_index[1]         # (M,)
        M   = src.shape[0]
        H_res = H                   # residual

        # ── multi-head projections ────────────────────────────────────────
        Q = self.W_Q(H).view(B, N, self.H, self.dk)    # (B, N, H, dk)
        K = self.W_K(H).view(B, N, self.H, self.dk)
        V = self.W_V(H).view(B, N, self.H, self.dk)

        # edge projections  → (M, H, dk)
        Ke = self.W_Ke(edge_attr).view(M, self.H, self.dk)
        Ve = self.W_Ve(edge_attr).view(M, self.H, self.dk)

        # ── gather per-edge query/key/value ───────────────────────────────
        Q_dst = Q[:, dst]                        # (B, M, H, dk)  query at dst
        K_src = K[:, src] + Ke.unsqueeze(0)     # (B, M, H, dk)  key at src + edge
        V_src = V[:, src] + Ve.unsqueeze(0)     # (B, M, H, dk)  val at src + edge

        # ── attention scores & segment softmax ────────────────────────────
        scores = (Q_dst * K_src).sum(-1) / math.sqrt(self.dk)  # (B, M, H)
        scores = scores.clamp(-10.0, 10.0)
        attn   = self._segment_softmax(scores, dst, N)          # (B, M, H)
        attn   = self.drop(attn)

        # ── weighted aggregation → destination nodes ──────────────────────
        weighted_V = attn.unsqueeze(-1) * V_src           # (B, M, H, dk)
        out = torch.zeros(B, N, self.H, self.dk,
                          device=H.device, dtype=H.dtype)
        idx_e = dst.view(1, M, 1, 1).expand(B, M, self.H, self.dk)
        out.scatter_add_(1, idx_e, weighted_V)
        out = out.reshape(B, N, self.d)
        out = self.W_O(out)

        # ── Add & Norm ────────────────────────────────────────────────────
        H = self.norm1(H_res + self.drop(out))
        H = self.norm2(H     + self.drop(self.ff(H)))
        return H


# ── §8b  Temporal Transformer Layer ─────────────────────────────────────────
class TemporalTransformerLayer(nn.Module):
    """
    Standard Transformer encoder layer applied along the time axis.

    After the graph transformer has produced spatially-enriched node
    embeddings for every timestep, this layer lets each node attend over
    its own W-length temporal history to capture how weather conditions
    (and spatial context) evolve before the prediction horizon.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, n_heads,
                                           dropout=dropout, batch_first=True)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (*, W, d_model)
        a, _ = self.attn(x, x, x)
        x = self.norm1(x + self.drop(a))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


# ── §8c  Full Spatiotemporal Model ───────────────────────────────────────────
class SpatioTemporalCloudTransformer(nn.Module):
    """
    End-to-end spatiotemporal graph transformer.

    Forward pass (all W timesteps in a single parallel call):

      1. NodeEmbedding
           Reshape (B, W, N, F) → (B·W, N, F)
           Linear + LayerNorm + GELU  → (B·W, N, d_model)

      2. Graph Transformer  [spatial — captures which neighbours matter]
           All B·W temporal slices are processed in one batched call.
           GraphTransformerLayer × n_gl  → (B·W, N, d_model)
           Reshape → (B, W, N, d_model)

      3. Temporal Transformer  [temporal — captures how things evolve]
           Transpose → (B, N, W, d_model)
           Add learnable positional encoding  (1, 1, W, d_model)
           Reshape → (B·N, W, d_model)
           TemporalTransformerLayer × n_tl  → (B·N, W, d_model)
           Take last token  → (B·N, d_model) → (B, N, d_model)

      4. Prediction head
           LayerNorm → Linear(d, d/2) → GELU → Linear(d/2, 1)
           → (B, N)   cloud cover at T+1
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_gl: int,
        n_tl: int,
        dropout: float,
        window: int,
    ):
        super().__init__()
        self.d      = d_model
        self.window = window

        # ── node embedding ────────────────────────────────────────────────
        self.node_embed = nn.Sequential(
            nn.Linear(node_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # ── spatial layers ────────────────────────────────────────────────
        self.graph_layers = nn.ModuleList([
            GraphTransformerLayer(d_model, n_heads, edge_dim, d_ff, dropout)
            for _ in range(n_gl)
        ])

        # ── temporal positional encoding  (learnable) ─────────────────────
        self.temp_pos = nn.Parameter(torch.zeros(1, 1, window, d_model))
        nn.init.trunc_normal_(self.temp_pos, std=0.02)

        # ── temporal layers ───────────────────────────────────────────────
        self.temp_layers = nn.ModuleList([
            TemporalTransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_tl)
        ])

        # ── prediction head ───────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        X_seq: torch.Tensor,         # (B, W, N, F)
        edge_index: torch.Tensor,    # (2, M)  long
        edge_attr: torch.Tensor,     # (M, 3)  float
    ) -> torch.Tensor:               # (B, N)

        B, W, N, _ = X_seq.shape

        # ── 1. Node embedding ─────────────────────────────────────────────
        H = self.node_embed(X_seq.view(B * W, N, -1))   # (B·W, N, d)

        # ── 2. Graph Transformer (all W timesteps in one batched call) ────
        for layer in self.graph_layers:
            H = layer(H, edge_index, edge_attr)           # (B·W, N, d)
        H = H.view(B, W, N, self.d)                       # (B, W, N, d)

        # ── 3. Temporal Transformer (per-node, across W timesteps) ────────
        H = H.permute(0, 2, 1, 3)                         # (B, N, W, d)
        H = H + self.temp_pos                              # learnable pos enc
        H = H.reshape(B * N, W, self.d)                   # (B·N, W, d)
        for layer in self.temp_layers:
            H = layer(H)
        H_last = H[:, -1, :].view(B, N, self.d)           # (B, N, d) — last token

        # ── 4. Prediction ─────────────────────────────────────────────────
        return self.head(H_last).squeeze(-1)               # (B, N)


# ═══════════════════════════════════════════════════════════════════════════
# §9  TRAINING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ei: torch.Tensor,
    ea: torch.Tensor,
    device: str,
) -> float:
    model.train()
    total = 0.0
    for X_seq, y in loader:
        X_seq, y = X_seq.to(device), y.to(device)
        loss = F.mse_loss(model(X_seq, ei, ea), y)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    ei: torch.Tensor,
    ea: torch.Tensor,
    device: str,
) -> tuple:
    """Returns (MSE, MAE, RMSE) on the normalised scale."""
    model.eval()
    preds, truths = [], []
    for X_seq, y in loader:
        preds.append(model(X_seq.to(device), ei, ea).cpu())
        truths.append(y)
    p = torch.cat(preds)
    t = torch.cat(truths)
    mse  = F.mse_loss(p, t).item()
    mae  = F.l1_loss(p, t).item()
    return mse, mae, math.sqrt(mse)


def save_plots(
    tr_losses: list,
    vl_losses: list,
    vl_rmse_pct: list,
    out_dir: str,
) -> None:
    """Save training-history plots to out_dir/training_history.png."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(tr_losses, label="Train MSE")
    axes[0].plot(vl_losses, label="Val MSE")
    axes[0].set(title="Training Loss", xlabel="Epoch", ylabel="MSE (normalised)")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(vl_rmse_pct, color="darkorange", label="Val RMSE")
    axes[1].set(title="Validation RMSE", xlabel="Epoch",
                ylabel="RMSE (% cloud cover)")
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    path = os.path.join(out_dir, "training_history.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {path}")


def save_prediction_scatter(
    model: nn.Module,
    loader: DataLoader,
    ei: torch.Tensor,
    ea: torch.Tensor,
    device: str,
    cloud_mean: float,
    cloud_std: float,
    out_dir: str,
    split: str = "test",
) -> None:
    """Scatter plot of predicted vs. actual cloud cover (% scale)."""
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for X_seq, y in loader:
            preds.append(model(X_seq.to(device), ei, ea).cpu())
            truths.append(y)
    p = torch.cat(preds).numpy().ravel() * cloud_std + cloud_mean
    t = torch.cat(truths).numpy().ravel() * cloud_std + cloud_mean

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(t, p, alpha=0.3, s=6, color="steelblue")
    lim = [0, 100]
    ax.plot(lim, lim, "r--", linewidth=1)
    ax.set(title=f"{split.title()} — Predicted vs Actual Cloud Cover",
           xlabel="Actual (%)", ylabel="Predicted (%)",
           xlim=lim, ylim=lim)
    ax.grid(True, alpha=0.4)
    rmse = math.sqrt(((p - t) ** 2).mean())
    mae  = np.abs(p - t).mean()
    ax.text(5, 88, f"RMSE = {rmse:.2f} %\nMAE  = {mae:.2f} %",
            fontsize=11, bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    path = os.path.join(out_dir, f"pred_vs_actual_{split}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Scatter plot saved → {path}")


def save_spatial_cloud_map(
    core_locs: pd.DataFrame,
    model: nn.Module,
    ds: "WeatherGraphDataset",
    ei: torch.Tensor,
    ea: torch.Tensor,
    device: str,
    cloud_mean: float,
    cloud_std: float,
    out_dir: str,
) -> None:
    """
    Map of predicted cloud cover at every node for the last test window.
    Circle size ∝ predicted cloud cover; colour = actual cloud cover.
    """
    model.eval()
    with torch.no_grad():
        X_last, y_last = ds[-1]                             # last window
        X_last = X_last.unsqueeze(0).to(device)             # (1, W, N, F)
        pred   = model(X_last, ei, ea).cpu().squeeze(0)     # (N,)

    pred_pct   = pred.numpy() * cloud_std + cloud_mean
    actual_pct = y_last.numpy() * cloud_std + cloud_mean

    lats = core_locs["latitude"].values
    lons = core_locs["longitude"].values

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, vals, title in zip(
        axes,
        [actual_pct, pred_pct],
        ["Actual Cloud Cover (%)", "Predicted Cloud Cover (%)"],
    ):
        sc = ax.scatter(lons, lats, c=vals, cmap="Blues",
                        s=vals * 2 + 5, vmin=0, vmax=100,
                        edgecolors="k", linewidths=0.3, alpha=0.85)
        plt.colorbar(sc, ax=ax, label="Cloud cover (%)")
        ax.set(title=title, xlabel="Longitude", ylabel="Latitude")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Spatial Cloud Cover — Last Test Window", fontsize=13)
    plt.tight_layout()
    path = os.path.join(out_dir, "spatial_cloud_map.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Spatial map saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════
# §10  MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    cfg = Cfg()
    os.makedirs(cfg.OUT_DIR, exist_ok=True)

    print("=" * 68)
    print("  Spatiotemporal Cloud Cover Graph Transformer")
    print(f"  Device : {cfg.DEVICE}")
    print("=" * 68)

    # ── §10.1  Data pipeline ───────────────────────────────────────────────
    print("\n[1/8]  Preprocessing …")
    df = preprocess(cfg.CSV_PATH)

    print("\n[2/8]  Finding core locations …")
    core_locs, dense_ts = get_core_locations(df, cfg.MIN_NODES_PER_TS, cfg.CORE_THRESH)

    print("\n[3/8]  Building KNN graph …")
    edge_index_np, edge_attr_np = build_knn_graph(core_locs, cfg.KNN_K)

    print("\n[4/8]  Building node-feature tensors …")
    X_all, y_all, feat_cols, full_ts = build_tensors(
        df, core_locs, dense_ts, target="cloud"
    )
    T, N, F_dim = X_all.shape

    # ── §10.2  Normalisation (fit ONLY on training portion) ────────────────
    print("\n[5/8]  Normalising …")
    n_tr  = int(T * cfg.TRAIN_F)
    n_val = int(T * cfg.VAL_F)

    scaler   = StandardScaler()
    scaler.fit(X_all[:n_tr].reshape(-1, F_dim))
    X_sc     = scaler.transform(X_all.reshape(-1, F_dim)).reshape(T, N, F_dim).astype(np.float32)

    # cloud scaler stats for inverse-transforming predictions → % scale
    cloud_idx  = feat_cols.index("cloud")
    cloud_mean = float(scaler.mean_[cloud_idx])
    cloud_std  = float(math.sqrt(scaler.var_[cloud_idx]))
    y_sc       = ((y_all - cloud_mean) / cloud_std).astype(np.float32)

    print(f"  Cloud cover stats  — mean: {cloud_mean:.2f} %, std: {cloud_std:.2f} %")

    # ── §10.3  Datasets & loaders ──────────────────────────────────────────
    W = cfg.WINDOW
    # Temporal split — no future leakage
    ds_tr  = WeatherGraphDataset(X_sc[:n_tr],             y_sc[:n_tr],             W)
    ds_val = WeatherGraphDataset(X_sc[n_tr:n_tr + n_val], y_sc[n_tr:n_tr + n_val], W)
    ds_te  = WeatherGraphDataset(X_sc[n_tr + n_val:],     y_sc[n_tr + n_val:],     W)

    dl_tr  = DataLoader(ds_tr,  batch_size=cfg.BATCH, shuffle=True,  drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=cfg.BATCH, shuffle=False)
    dl_te  = DataLoader(ds_te,  batch_size=cfg.BATCH, shuffle=False)

    print(f"\n  Windows  →  Train: {len(ds_tr)}  |  Val: {len(ds_val)}  |  Test: {len(ds_te)}")

    # ── §10.4  Model ───────────────────────────────────────────────────────
    print("\n[6/8]  Building model …")
    edge_index = torch.from_numpy(edge_index_np).long().to(cfg.DEVICE)
    edge_attr  = torch.from_numpy(edge_attr_np).float().to(cfg.DEVICE)

    model = SpatioTemporalCloudTransformer(
        node_dim = F_dim,
        edge_dim = 3,
        d_model  = cfg.D_MODEL,
        n_heads  = cfg.N_HEADS,
        d_ff     = cfg.D_MODEL * 4,
        n_gl     = cfg.N_GL,
        n_tl     = cfg.N_TL,
        dropout  = cfg.DROPOUT,
        window   = W,
    ).to(cfg.DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  N={N} nodes  |  F={F_dim} features  |  M={edge_index_np.shape[1]} edges  "
          f"|  Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.LR, weight_decay=cfg.WD
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.EPOCHS, eta_min=cfg.LR * 0.01
    )

    # ── §10.5  Training loop ───────────────────────────────────────────────
    print("\n[7/8]  Training …")
    tr_losses, vl_losses, vl_rmse_pct = [], [], []
    best_val, patience_ctr, best_state = float("inf"), 0, None

    for ep in range(1, cfg.EPOCHS + 1):
        tr_l              = train_one_epoch(model, dl_tr, optimizer,
                                            edge_index, edge_attr, cfg.DEVICE)
        v_mse, v_mae, v_rmse = evaluate(model, dl_val,
                                         edge_index, edge_attr, cfg.DEVICE)
        scheduler.step()

        tr_losses.append(tr_l)
        vl_losses.append(v_mse)
        vl_rmse_pct.append(v_rmse * cloud_std)

        # ── early stopping ───────────────────────────────────────────────
        if v_mse < best_val:
            best_val    = v_mse
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
        if patience_ctr >= cfg.PATIENCE:
            print(f"\n  Early stopping at epoch {ep} (no val improvement for {cfg.PATIENCE} epochs).")
            break

        if ep % 5 == 0 or ep == 1:
            print(
                f"  Ep {ep:3d}/{cfg.EPOCHS}  "
                f"train_MSE={tr_l:.4f}  "
                f"val_MSE={v_mse:.4f}  "
                f"val_RMSE={v_rmse * cloud_std:.2f}%  "
                f"val_MAE={v_mae * cloud_std:.2f}%  "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

    # ── §10.6  Test evaluation ─────────────────────────────────────────────
    print("\n[8/8]  Testing (best checkpoint) …")
    model.load_state_dict({k: v.to(cfg.DEVICE) for k, v in best_state.items()})
    t_mse, t_mae, t_rmse = evaluate(model, dl_te, edge_index, edge_attr, cfg.DEVICE)

    print("\n" + "─" * 50)
    print("  FINAL TEST RESULTS")
    print(f"    RMSE : {t_rmse * cloud_std:.4f} % cloud cover")
    print(f"    MAE  : {t_mae  * cloud_std:.4f} % cloud cover")
    print(f"    MSE  : {t_mse:.6f}  (normalised scale)")
    print("─" * 50)

    # ── §10.7  Save artefacts ──────────────────────────────────────────────
    save_plots(tr_losses, vl_losses, vl_rmse_pct, cfg.OUT_DIR)
    save_prediction_scatter(
        model, dl_te, edge_index, edge_attr, cfg.DEVICE,
        cloud_mean, cloud_std, cfg.OUT_DIR, split="test"
    )
    save_spatial_cloud_map(
        core_locs, model, ds_te,
        edge_index, edge_attr, cfg.DEVICE,
        cloud_mean, cloud_std, cfg.OUT_DIR
    )

    torch.save(
        {
            "model_state" : best_state,
            "feat_cols"   : feat_cols,
            "core_locs"   : core_locs,
            "cloud_mean"  : cloud_mean,
            "cloud_std"   : cloud_std,
            "edge_index"  : edge_index_np,
            "edge_attr"   : edge_attr_np,
            "scaler_mean" : scaler.mean_,
            "scaler_var"  : scaler.var_,
            "cfg"         : cfg.__dict__ if hasattr(cfg, "__dict__") else {},
            "test_rmse"   : t_rmse * cloud_std,
            "test_mae"    : t_mae  * cloud_std,
        },
        os.path.join(cfg.OUT_DIR, "cloud_model.pt"),
    )
    print(f"  Model saved  → {cfg.OUT_DIR}/cloud_model.pt")

    print("\n" + "=" * 68)
    print("  Done.")
    print("=" * 68)


if __name__ == "__main__":
    main()