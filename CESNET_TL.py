#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import time
import copy
import random
import gc
import multiprocessing
import numpy as np
import pandas as pd
import joblib
from scipy.stats import entropy as shannon_entropy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from cesnet_datazoo.datasets import CESNET_QUIC22
from cesnet_datazoo.config import DatasetConfig, AppSelection

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

n_cores = multiprocessing.cpu_count()
torch.set_num_threads(n_cores)
print(f"Using {n_cores} CPU threads for PyTorch")

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

set_seed(42)


# In[2]:


DATA_DIR = "/datasets/CESNET-QUIC22/"
MAX_SEQ_LEN = 30
MIN_PACKETS = 5
LARGE_PKT_THRESHOLD = 1000
SMALL_PKT_THRESHOLD = 100
CHUNK_SIZE = 100_000

dataset = CESNET_QUIC22(DATA_DIR, size="XS")
dataset_config = DatasetConfig(
    dataset=dataset,
    apps_selection=AppSelection.ALL_KNOWN,
    train_period_name="W-2022-44",
    test_period_name="W-2022-45",
)
dataset.set_dataset_config_and_initialize(dataset_config)
print("Dataset initialized for CESNET-QUIC22 (XS)")


# In[3]:


train_df = dataset.get_train_df()
val_df   = dataset.get_val_df()
test_df  = dataset.get_test_df()

print(f"Train: {len(train_df)} flows")
print(f"Val:   {len(val_df)} flows")
print(f"Test:  {len(test_df)} flows")
print(f"Classes in train: {train_df['APP'].nunique()}")

# check whether APP codes map to readable names
print("APP dtype:", train_df["APP"].dtype)
if hasattr(train_df["APP"], "cat"):
    print("Categories:", list(train_df["APP"].cat.categories))


# In[4]:


train_counts = train_df["APP"].value_counts()
val_counts = val_df["APP"].value_counts()
test_counts = test_df["APP"].value_counts()

all_apps = sorted(set(train_df["APP"]) | set(val_df["APP"]) | set(test_df["APP"]))

print(f"Total unique APP classes: {len(all_apps)}")
print("=" * 100)

for app in all_apps:
    train_app = train_df[train_df["APP"] == app]
    val_app   = val_df[val_df["APP"] == app]
    test_app  = test_df[test_df["APP"] == app]
    print(f"\nAPP: {app}")
    print(f"  Train: {len(train_app)} flows")
    print(f"  Val:   {len(val_app)} flows")
    print(f"  Test:  {len(test_app)} flows")
    print(f"  Total: {len(train_app) + len(val_app) + len(test_app)} flows")


# In[5]:


chosen_apps = [31, 39, 53,3,4]

train_counts = train_df["APP"].value_counts()
val_counts = val_df["APP"].value_counts()
test_counts = test_df["APP"].value_counts()

print("Selected APP codes:", chosen_apps)
print("\nCounts per split for chosen classes:")
for app in chosen_apps:
    print(f"  APP {app}: train={train_counts.get(app,0)}, "
          f"val={val_counts.get(app,0)}, test={test_counts.get(app,0)}")

train_df = train_df[train_df["APP"].isin(chosen_apps)].reset_index(drop=True)
val_df   = val_df[val_df["APP"].isin(chosen_apps)].reset_index(drop=True)
test_df  = test_df[test_df["APP"].isin(chosen_apps)].reset_index(drop=True)

print(f"\nAfter filtering to APP {chosen_apps}:")
print(f"Train: {len(train_df)} flows")
print(f"Val:   {len(val_df)} flows")
print(f"Test:  {len(test_df)} flows")


# In[6]:


def stack_ppi(df):
    ppi_stack = np.stack(df["PPI"].values)
    if ppi_stack.shape[1] == 3:
        sizes, directions, ipt = ppi_stack[:, 0, :], ppi_stack[:, 1, :], ppi_stack[:, 2, :]
    else:
        sizes, directions, ipt = ppi_stack[:, :, 0], ppi_stack[:, :, 1], ppi_stack[:, :, 2]

    sizes = sizes.astype(np.float32)
    directions = directions.astype(np.float32)
    ipt = ipt.astype(np.float32)

    valid_mask = (sizes >= 0) & (ipt >= 0)

    real_dirs = directions[valid_mask]
    uniq = set(np.unique(real_dirs).tolist()) if real_dirs.size else set()
    if uniq <= {-1.0, 1.0}:
        directions = np.where(valid_mask, (directions > 0).astype(np.float32), 0.0)
    elif uniq <= {0.0, 1.0}:
        directions = np.where(valid_mask, directions, 0.0)
    else:
        med = np.median(real_dirs) if real_dirs.size else 0.0
        directions = np.where(valid_mask, (directions > med).astype(np.float32), 0.0)

    sizes = np.clip(np.where(valid_mask, sizes, 0.0), 0, None)
    ipt = np.clip(np.where(valid_mask, ipt, 0.0), 0, None)
    cum_time = np.cumsum(ipt, axis=1)

    return sizes, directions, ipt, cum_time, valid_mask

def longest_run_vectorized(mask2d):
    N, L = mask2d.shape
    cur = np.zeros(N, dtype=np.int32)
    best = np.zeros(N, dtype=np.int32)
    for j in range(L):
        col = mask2d[:, j]
        cur = np.where(col, cur + 1, 0)
        best = np.maximum(best, cur)
    return best

def avg_run_vectorized(mask2d):
    N, L = mask2d.shape
    cur = np.zeros(N, dtype=np.int64)
    run_sum = np.zeros(N, dtype=np.float64)
    run_count = np.zeros(N, dtype=np.int64)
    for j in range(L):
        col = mask2d[:, j]
        ending = (~col) & (cur > 0)
        run_sum[ending] += cur[ending]
        run_count[ending] += 1
        cur = np.where(col, cur + 1, 0)
    ending = cur > 0
    run_sum[ending] += cur[ending]
    run_count[ending] += 1
    return np.divide(run_sum, run_count, out=np.zeros_like(run_sum), where=run_count > 0)

def masked_stat(values, mask, fn, fill=0.0):
    tmp = np.where(mask, values, np.nan)
    with np.errstate(all='ignore'):
        out = fn(tmp, axis=1)
    return np.nan_to_num(out, nan=fill)


# In[7]:


def build_arrays_vectorized(df):
    sizes, directions, ipt, cum_time, seq_mask = stack_ppi(df)
    N, L = sizes.shape

    real_len = seq_mask.sum(axis=1)
    keep = real_len >= MIN_PACKETS

    sizes, directions, ipt, cum_time, seq_mask = \
        sizes[keep], directions[keep], ipt[keep], cum_time[keep], seq_mask[keep]
    real_len = real_len[keep]
    labels = df["APP"].values[keep]

    log_size = np.log1p(sizes)
    log_iat = np.log1p(ipt)
    X_seq = np.stack([log_size, directions, log_iat, cum_time], axis=2).astype(np.float32)
    X_seq = X_seq * seq_mask[:, :, None]

    up = seq_mask & (directions == 1)
    down = seq_mask & (directions == 0)
    iats = np.diff(np.where(seq_mask, cum_time, np.nan), axis=1, prepend=cum_time[:, :1])

    stat_df = pd.DataFrame({
        "total_pkts": real_len,
        "total_bytes": sizes.sum(axis=1),
        "flow_duration": np.clip(
            masked_stat(cum_time, seq_mask, np.nanmax) - masked_stat(cum_time, seq_mask, np.nanmin),
            1e-6, None),
        "mean_pkt_size": masked_stat(sizes, seq_mask, np.nanmean),
        "std_pkt_size": masked_stat(sizes, seq_mask, np.nanstd),
        "min_pkt_size": masked_stat(sizes, seq_mask, np.nanmin),
        "max_pkt_size": masked_stat(sizes, seq_mask, np.nanmax),
        "mean_iat": np.nan_to_num(np.nanmean(iats, axis=1)),
        "std_iat": np.nan_to_num(np.nanstd(iats, axis=1)),
        "pkts_up": up.sum(axis=1), "pkts_down": down.sum(axis=1),
        "bytes_up": np.where(up, sizes, 0).sum(axis=1),
        "bytes_down": np.where(down, sizes, 0).sum(axis=1),
    })

    large = seq_mask & (sizes >= LARGE_PKT_THRESHOLD)
    small = seq_mask & (sizes <= SMALL_PKT_THRESHOLD)
    iat_pos = np.where(iats > 0, iats, np.nan)

    burst_df = pd.DataFrame({
        "burst_up_total": up.sum(axis=1), "burst_down_total": down.sum(axis=1),
        "burst_up_longest": longest_run_vectorized(up), "burst_down_longest": longest_run_vectorized(down),
        "burst_up_avg": avg_run_vectorized(up), "burst_down_avg": avg_run_vectorized(down),
        "large_pkt_count": large.sum(axis=1), "small_pkt_count": small.sum(axis=1),
        "large_up": (up & large).sum(axis=1), "large_down": (down & large).sum(axis=1),
        "small_up": (up & small).sum(axis=1), "small_down": (down & small).sum(axis=1),
        "iat_max": np.nan_to_num(np.nanmax(iat_pos, axis=1), nan=1e-6),
        "iat_min": np.nan_to_num(np.nanmin(iat_pos, axis=1), nan=1e-6),
        "iat_mean": np.nan_to_num(np.nanmean(iat_pos, axis=1), nan=1e-6),
        "iat_std": np.nan_to_num(np.nanstd(iat_pos, axis=1)),
    })
    denom = sizes.sum(axis=1, keepdims=True)
    probs = np.divide(sizes, denom, out=np.zeros_like(sizes), where=denom > 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        ent_terms = np.where(probs > 0, probs * np.log2(probs), 0.0)
    burst_df["entropy"] = -ent_terms.sum(axis=1)

    return X_seq, real_len.astype(np.int64), stat_df, burst_df, labels

def build_arrays_chunked(df, chunk_size=CHUNK_SIZE):
    n_total = len(df)
    seq_chunks, len_chunks, stat_chunks, burst_chunks, label_chunks = [], [], [], [], []

    for start in range(0, n_total, chunk_size):
        end = min(start + chunk_size, n_total)
        chunk = df.iloc[start:end]
        X_seq_c, real_len_c, stat_df_c, burst_df_c, labels_c = build_arrays_vectorized(chunk)

        seq_chunks.append(X_seq_c)
        len_chunks.append(real_len_c)
        stat_chunks.append(stat_df_c)
        burst_chunks.append(burst_df_c)
        label_chunks.append(labels_c)

        print(f"  chunk {start:>8}-{end:<8} / {n_total}", end="\r")
        del chunk, X_seq_c, real_len_c, stat_df_c, burst_df_c, labels_c
        gc.collect()

    print()
    X_seq = np.concatenate(seq_chunks, axis=0)
    real_len = np.concatenate(len_chunks, axis=0)
    stat_df = pd.concat(stat_chunks, ignore_index=True)
    burst_df = pd.concat(burst_chunks, ignore_index=True)
    labels = np.concatenate(label_chunks, axis=0)
    return X_seq, real_len, stat_df, burst_df, labels

print("Processing train split...")
X_seq_train_raw, X_len_train_raw, X_stat_train_df, X_burst_train_df, y_train_raw = build_arrays_chunked(train_df)
del train_df; gc.collect()

print("Processing val split...")
X_seq_val_raw, X_len_val_raw, X_stat_val_df, X_burst_val_df, y_val_raw = build_arrays_chunked(val_df)
del val_df; gc.collect()

print("Processing test split...")
X_seq_test_raw, X_len_test_raw, X_stat_test_df, X_burst_test_df, y_test_raw = build_arrays_chunked(test_df)
del test_df; gc.collect()

print(f"\nFeature shapes (train):\n  X_seq: {X_seq_train_raw.shape}\n  X_stat: {X_stat_train_df.shape}\n  X_burst: {X_burst_train_df.shape}\n  y: {y_train_raw.shape}")

assert np.isfinite(X_seq_train_raw).all(), "Still non-finite values in X_seq_train_raw!"
assert np.isfinite(X_seq_val_raw).all(), "Still non-finite values in X_seq_val_raw!"
assert np.isfinite(X_seq_test_raw).all(), "Still non-finite values in X_seq_test_raw!"
print("Sanity check passed on all three splits: no inf/nan")


# In[8]:


X_seq_train = X_seq_train_raw.astype(np.float32)
X_seq_val   = X_seq_val_raw.astype(np.float32)
X_seq_test  = X_seq_test_raw.astype(np.float32)

seq_scaler = StandardScaler()
seq_scaler.fit(X_seq_train.reshape(-1, X_seq_train.shape[-1]))

def scale_seq(X):
    shp = X.shape
    return seq_scaler.transform(X.reshape(-1, shp[-1])).reshape(shp)

X_seq_train = scale_seq(X_seq_train)
X_seq_val   = scale_seq(X_seq_val)
X_seq_test  = scale_seq(X_seq_test)

stat_scaler = StandardScaler().fit(X_stat_train_df)
X_stat_train = stat_scaler.transform(X_stat_train_df)
X_stat_val   = stat_scaler.transform(X_stat_val_df)
X_stat_test  = stat_scaler.transform(X_stat_test_df)

burst_scaler = StandardScaler().fit(X_burst_train_df)
X_burst_train = burst_scaler.transform(X_burst_train_df)
X_burst_val   = burst_scaler.transform(X_burst_val_df)
X_burst_test  = burst_scaler.transform(X_burst_test_df)

le = LabelEncoder()
le.fit(y_train_raw)
n_classes = len(le.classes_)
print(f"Classes: {n_classes} -> {le.classes_}")

def encode_and_filter(y_raw, X_seq, X_stat, X_burst, X_len):
    mask = np.isin(y_raw, le.classes_)
    dropped = (~mask).sum()
    if dropped:
        print(f"  Dropping {dropped} rows with unseen classes")
    y_enc = le.transform(y_raw[mask])
    return X_seq[mask], X_stat[mask], X_burst[mask], X_len[mask], y_enc

print("Filtering train...")
X_seq_train, X_stat_train, X_burst_train, len_train, y_train = \
    encode_and_filter(y_train_raw, X_seq_train, X_stat_train, X_burst_train, X_len_train_raw)
print("Filtering val...")
X_seq_val, X_stat_val, X_burst_val, len_val, y_val = \
    encode_and_filter(y_val_raw, X_seq_val, X_stat_val, X_burst_val, X_len_val_raw)
print("Filtering test...")
X_seq_test, X_stat_test, X_burst_test, len_test, y_test = \
    encode_and_filter(y_test_raw, X_seq_test, X_stat_test, X_burst_test, X_len_test_raw)

print(f"\nSplit sizes - Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")


# In[9]:


class QUICMultiDataset(Dataset):
    def __init__(self, X_seq, X_stat, X_burst, lengths, y):
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.X_stat = torch.tensor(X_stat, dtype=torch.float32)
        self.X_burst = torch.tensor(X_burst, dtype=torch.float32)
        self.lengths = torch.tensor(lengths, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_seq[idx], self.X_stat[idx], self.X_burst[idx], self.lengths[idx], self.y[idx]

def make_loaders(batch_size=256):
    train_ds = QUICMultiDataset(X_seq_train, X_stat_train, X_burst_train, len_train, y_train)
    val_ds   = QUICMultiDataset(X_seq_val, X_stat_val, X_burst_val, len_val, y_val)
    test_ds  = QUICMultiDataset(X_seq_test, X_stat_test, X_burst_test, len_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

train_loader, val_loader, test_loader = make_loaders(batch_size=256)
print("DataLoaders created (CPU-tuned batch size=256)")


# In[10]:


class MultiHeadAdditiveAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, attention_dropout=0.2):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.temperature = nn.Parameter(torch.ones(1) * (self.head_dim ** -0.5))
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.attention_weights = None

    def forward(self, x, lengths):
        batch_size, seq_len, hidden_dim = x.size()
        x_norm = self.norm(x)
        Q = self.query_proj(x_norm)
        K = self.key_proj(x_norm)
        V = self.value_proj(x)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.temperature
        mask = torch.arange(seq_len, device=x.device)[None, :] < lengths[:, None]
        mask = mask.unsqueeze(1).unsqueeze(1)
        scores = scores.masked_fill(~mask, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        self.attention_weights = weights.detach()
        weights = self.attn_dropout(weights)
        attn_output = torch.matmul(weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, hidden_dim)
        attn_output = self.output_proj(attn_output)
        context = attn_output.sum(dim=1)
        context = context / lengths.unsqueeze(-1).float().clamp(min=1)
        avg_weights = weights.mean(dim=1).max(dim=1)[0]
        avg_weights = avg_weights.masked_fill(~mask.squeeze(1).squeeze(1), 0.0)
        weight_sum = avg_weights.sum(dim=1, keepdim=True).clamp(min=1e-9)
        avg_weights = avg_weights / weight_sum
        return context, avg_weights


# In[11]:


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, max(channels // reduction, 4))
        self.fc2 = nn.Linear(max(channels // reduction, 4), channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch, channels, seq_len = x.size()
        se = x.mean(dim=2)
        se = self.relu(self.fc1(se))
        se = self.sigmoid(self.fc2(se))
        return x * se.unsqueeze(2)

class CNNBiGRUAttnEncoder(nn.Module):
    def __init__(self, input_dim=4, cnn_channels=48, gru_hidden=96, num_gru_layers=2,
                 dropout=0.2, use_residual=True):
        super().__init__()
        self.input_dim = input_dim
        self.cnn_channels = cnn_channels
        self.gru_hidden = gru_hidden
        self.use_residual = use_residual

        self.conv1_3x3 = nn.Conv1d(input_dim, cnn_channels, kernel_size=3, padding=1, bias=False)
        self.conv1_5x5 = nn.Conv1d(input_dim, cnn_channels, kernel_size=5, padding=2, bias=False)
        self.conv1_7x7 = nn.Conv1d(input_dim, cnn_channels, kernel_size=7, padding=3, bias=False)

        self.bn1 = nn.BatchNorm1d(cnn_channels * 3)
        self.se1 = SEBlock(cnn_channels * 3)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(cnn_channels * 3, cnn_channels * 3, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(cnn_channels * 3)
        self.se2 = SEBlock(cnn_channels * 3)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.conv3 = nn.Conv1d(cnn_channels * 3, cnn_channels * 4, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(cnn_channels * 4)
        self.se3 = SEBlock(cnn_channels * 4)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)

        self.res_proj1 = nn.Conv1d(input_dim, cnn_channels * 3, kernel_size=1, bias=False) if use_residual else None

        self.bigru = nn.GRU(
            input_size=cnn_channels * 4,
            hidden_size=gru_hidden,
            num_layers=num_gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_gru_layers > 1 else 0.0
        )

        self.gru_norm = nn.LayerNorm(gru_hidden * 2)
        self.attention = MultiHeadAdditiveAttention(gru_hidden * 2, num_heads=6, attention_dropout=dropout)
        self.out_dim = gru_hidden * 2
        self.last_attn_weights = None
        self.cnn_features = None

    def forward(self, x, lengths):
        x = x.permute(0, 2, 1)
        h1_3x3 = self.conv1_3x3(x)
        h1_5x5 = self.conv1_5x5(x)
        h1_7x7 = self.conv1_7x7(x)
        h1 = torch.cat([h1_3x3, h1_5x5, h1_7x7], dim=1)
        h1 = self.bn1(h1)
        h1 = self.se1(h1)
        h1 = self.relu1(h1)
        h1 = self.dropout1(h1)

        if self.use_residual and self.res_proj1 is not None:
            x_proj = self.res_proj1(x)
            h1 = h1 + x_proj

        h2 = self.conv2(h1)
        h2 = self.bn2(h2)
        h2 = self.se2(h2)
        h2 = self.relu2(h2)
        h2 = self.dropout2(h2)
        if self.use_residual:
            h2 = h2 + h1

        h3 = self.conv3(h2)
        h3 = self.bn3(h3)
        h3 = self.se3(h3)
        h3 = self.relu3(h3)
        h3 = self.dropout3(h3)

        self.cnn_features = h3.detach()
        h3 = h3.permute(0, 2, 1)

        lengths_cpu = lengths.clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(h3, lengths_cpu, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.bigru(packed)
        h_gru, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=h3.size(1))

        h_gru = self.gru_norm(h_gru)
        context, weights = self.attention(h_gru, lengths)
        self.last_attn_weights = weights.detach()
        return context


# In[12]:


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        p = F.softmax(inputs, dim=1)
        class_mask = F.one_hot(targets, num_classes=inputs.size(1))
        probs = (p * class_mask).sum(dim=1)
        focal_weight = (1 - probs) ** self.gamma
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        focal_loss = focal_weight * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha if isinstance(self.alpha, (float, int)) else self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, num_classes, smoothing=0.1, weight=None):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.register_buffer('weight', weight)
        self.confidence = 1.0 - smoothing
        self.smooth_value = smoothing / num_classes

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smooth_value)
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        loss = torch.sum(-true_dist * log_probs, dim=1)
        if self.weight is not None:
            loss = loss * self.weight[targets]
        return loss.mean()

def compute_class_weights(train_loader, num_classes, device='cpu'):
    class_counts = torch.zeros(num_classes, device=device)
    for _, _, _, _, y_batch in train_loader:
        for y in y_batch:
            class_counts[y.item()] += 1
    total_samples = class_counts.sum()
    class_weights = total_samples / (num_classes * class_counts.clamp(min=1))
    class_weights = class_weights / class_weights.mean()
    return class_weights


# In[13]:


def train_with_early_stopping(model, train_loader, val_loader, max_epochs=40,
                             learning_rate=1e-3, patience=10, verbose=True,
                             compute_class_weights=True, use_ema=True, ema_decay=0.999):
    device = next(model.parameters()).device
    class_counts = {}
    if compute_class_weights:
        for _, _, _, _, y_batch in train_loader:
            for y in y_batch.cpu().numpy():
                class_counts[y] = class_counts.get(y, 0) + 1
        total_samples = sum(class_counts.values())
        n_classes_ = max(class_counts.keys()) + 1
        class_weights = torch.ones(n_classes_, device=device)
        for class_id, count in class_counts.items():
            class_weights[class_id] = total_samples / (n_classes_ * count)
        if verbose:
            print(f"Class weights: {class_weights}")
    else:
        class_weights = None

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    def lr_lambda(epoch):
        warmup_epochs = 5
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    class EMA:
        def __init__(self, model, decay=0.999):
            self.model = model
            self.decay = decay
            self.shadow = {}
            self.backup = {}
            self.register()
        def register(self):
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.shadow[name] = param.data.clone()
        def update(self):
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    ema = EMA(model, decay=ema_decay) if use_ema else None

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": [], "learning_rate": []}
    best_val_f1 = 0.0
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        for x_seq, x_stat, x_burst, seq_len, y_batch in train_loader:
            x_seq, x_stat, x_burst = x_seq.to(device), x_stat.to(device), x_burst.to(device)
            seq_len, y_batch = seq_len.to(device), y_batch.to(device)
            out = model(x_seq, x_stat, x_burst, seq_len)
            loss = criterion(out, y_batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if use_ema:
                ema.update()
            total_loss += loss.item() * x_seq.size(0)

        train_loss = total_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for x_seq, x_stat, x_burst, seq_len, y_batch in val_loader:
                x_seq, x_stat, x_burst = x_seq.to(device), x_stat.to(device), x_burst.to(device)
                seq_len, y_batch = seq_len.to(device), y_batch.to(device)
                out = model(x_seq, x_stat, x_burst, seq_len)
                loss = criterion(out, y_batch)
                val_loss += loss.item() * x_seq.size(0)
                all_preds.append(out.argmax(dim=1).cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        val_acc = np.mean(all_preds == all_targets)
        val_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])

        if verbose and (epoch % 5 == 0 or epoch == max_epochs - 1):
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch+1:2d}/{max_epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | val_f1={val_f1:.4f} | lr={lr_now:.2e}")

        if val_f1 > best_val_f1 + 1e-4:
            best_val_f1, best_val_loss, best_val_acc = val_f1, val_loss, val_acc
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            if verbose:
                print(f"  Best F1: {best_val_f1:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    best_val_metrics = {"f1": best_val_f1, "accuracy": best_val_acc, "loss": best_val_loss}
    return model, history, best_val_metrics


# In[14]:


class MultiBranchNet(nn.Module):
    def __init__(self, stat_dim, burst_dim, n_classes,
                 use_seq=True, use_stat=True, use_burst=True,
                 dropout=0.2, stat_hidden=64, burst_hidden=64,
                 use_residual_fusion=True,
                 seq_cnn_channels=48, seq_gru_hidden=96, seq_num_gru_layers=2):
        super().__init__()
        self.use_seq = use_seq
        self.use_stat = use_stat
        self.use_burst = use_burst
        self.n_branches = sum([use_seq, use_stat, use_burst])
        self.use_residual_fusion = use_residual_fusion
        fusion_dim = 0

        if use_seq:
            self.seq_encoder = CNNBiGRUAttnEncoder(
                cnn_channels=seq_cnn_channels,
                gru_hidden=seq_gru_hidden,
                num_gru_layers=seq_num_gru_layers,
                dropout=dropout,
            )
            self.seq_output_dim = self.seq_encoder.out_dim
            self.seq_norm = nn.LayerNorm(self.seq_output_dim)
            self.seq_pre_fusion = nn.Sequential(
                nn.Linear(self.seq_output_dim, self.seq_output_dim),
                nn.LayerNorm(self.seq_output_dim),
                nn.ReLU(),
            )
            fusion_dim += self.seq_output_dim

        if use_stat:
            self.stat_mlp = nn.Sequential(
                nn.Linear(stat_dim, stat_hidden), nn.BatchNorm1d(stat_hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(stat_hidden, stat_hidden), nn.BatchNorm1d(stat_hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(stat_hidden, stat_hidden), nn.ReLU(),
            )
            self.stat_norm = nn.LayerNorm(stat_hidden)
            self.stat_pre_fusion = nn.Sequential(
                nn.Linear(stat_hidden, stat_hidden), nn.LayerNorm(stat_hidden), nn.ReLU(),
            )
            fusion_dim += stat_hidden

        if use_burst:
            self.burst_mlp = nn.Sequential(
                nn.Linear(burst_dim, burst_hidden), nn.BatchNorm1d(burst_hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(burst_hidden, burst_hidden), nn.BatchNorm1d(burst_hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(burst_hidden, burst_hidden), nn.ReLU(),
            )
            self.burst_norm = nn.LayerNorm(burst_hidden)
            self.burst_pre_fusion = nn.Sequential(
                nn.Linear(burst_hidden, burst_hidden), nn.LayerNorm(burst_hidden), nn.ReLU(),
            )
            fusion_dim += burst_hidden

        self.branch_gate = nn.Sequential(
            nn.Linear(fusion_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, self.n_branches), nn.Softmax(dim=1)
        )
        if use_residual_fusion:
            self.fusion_residual_proj = nn.Linear(fusion_dim, fusion_dim)
        self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x_seq, x_stat, x_burst, seq_len, return_intermediate=False):
        intermediate = {} if return_intermediate else None
        parts, branch_outputs = [], []

        if self.use_seq:
            seq_out = self.seq_pre_fusion(self.seq_norm(self.seq_encoder(x_seq, seq_len)))
            parts.append(seq_out); branch_outputs.append(seq_out)
            if return_intermediate:
                intermediate['seq_out'] = seq_out.detach()
                intermediate['attention_weights'] = self.seq_encoder.last_attn_weights

        if self.use_stat:
            stat_out = self.stat_pre_fusion(self.stat_norm(self.stat_mlp(x_stat)))
            parts.append(stat_out); branch_outputs.append(stat_out)
            if return_intermediate:
                intermediate['stat_out'] = stat_out.detach()

        if self.use_burst:
            burst_out = self.burst_pre_fusion(self.burst_norm(self.burst_mlp(x_burst)))
            parts.append(burst_out); branch_outputs.append(burst_out)
            if return_intermediate:
                intermediate['burst_out'] = burst_out.detach()

        fused = torch.cat(parts, dim=1)
        branch_weights = self.branch_gate(fused)
        if return_intermediate:
            intermediate['branch_weights'] = branch_weights.detach()

        weighted_parts = [b * branch_weights[:, i:i+1] for i, b in enumerate(branch_outputs)]
        fused_weighted = torch.cat(weighted_parts, dim=1)

        if self.use_residual_fusion:
            fused_weighted = fused_weighted + 0.5 * self.fusion_residual_proj(fused)

        logits = self.classifier(self.fusion_norm(fused_weighted))
        return (logits, intermediate) if return_intermediate else logits


# In[15]:


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_size_mb(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)

def evaluate_model(model, loader, class_names, label="Model", plot=True):
    model.eval()
    all_preds, all_labels = [], []
    inference_times = []
    with torch.no_grad():
        for x_seq, x_stat, x_burst, seq_len, y_batch in loader:
            x_seq, x_stat, x_burst, seq_len = x_seq.to(device), x_stat.to(device), x_burst.to(device), seq_len.to(device)
            start = time.time()
            out = model(x_seq, x_stat, x_burst, seq_len)
            inference_times.append((time.time() - start) / x_seq.size(0))
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(y_batch.numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    avg_latency = np.mean(inference_times) * 1000

    print(f"\n{label}")
    print(f"  F1-Score:  {f1:.6f}")
    print(f"  Accuracy:  {acc:.6f}")
    print(f"  Precision: {prec:.6f}")
    print(f"  Recall:    {rec:.6f}")
    print(f"  Latency:   {avg_latency:.4f} ms/sample")
    print(f"  Params:    {count_params(model):,}")
    print(f"  Size:      {model_size_mb(model):.4f} MB")

    if plot:
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title(f'{label} - Confusion Matrix')
        plt.ylabel('True'); plt.xlabel('Predicted')
        plt.tight_layout(); plt.show()

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "latency_ms": avg_latency}

def visualize_attention_weights(model, loader, class_names, n_samples=1):
    model.eval()
    with torch.no_grad():
        for batch_idx, (x_seq, x_stat, x_burst, seq_len, y_batch) in enumerate(loader):
            if batch_idx >= n_samples:
                break
            x_seq, x_stat, x_burst, seq_len = x_seq.to(device), x_stat.to(device), x_burst.to(device), seq_len.to(device)
            logits, intermediate = model(x_seq, x_stat, x_burst, seq_len, return_intermediate=True)
            attn_weights = intermediate['attention_weights']
            real_len = seq_len[0].item()
            plt.figure(figsize=(12, 4))
            plt.bar(range(real_len), attn_weights[0, :real_len].cpu().numpy(), color='skyblue', edgecolor='navy')
            plt.xlabel('Packet Position'); plt.ylabel('Attention Weight')
            pred_class = class_names[logits[0].argmax().item()]
            plt.title(f'Attention Weights - {pred_class}')
            plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

def compute_gradcam_packet_importance(model, x_seq, x_stat, x_burst, seq_len, target_class):
    model.eval()
    x_seq = x_seq.to(device).requires_grad_(True)
    x_stat, x_burst, seq_len = x_stat.to(device), x_burst.to(device), seq_len.to(device)
    logits = model(x_seq, x_stat, x_burst, seq_len)
    loss = logits[0, target_class]
    model.zero_grad(); loss.backward()
    gradients = x_seq.grad.data
    importance = gradients.abs().mean(dim=2)[0]
    real_len = seq_len[0].item()
    importance = importance[:real_len]
    importance_norm = (importance - importance.min()) / (importance.max() - importance.min() + 1e-9)
    return importance_norm.cpu().numpy()

def visualize_gradcam(model, loader, class_names, n_samples=1):
    model.eval()
    for batch_idx, (x_seq, x_stat, x_burst, seq_len, y_batch) in enumerate(loader):
        if batch_idx >= n_samples:
            break
        x_seq, x_stat, x_burst, seq_len = x_seq.to(device), x_stat.to(device), x_burst.to(device), seq_len.to(device)
        with torch.no_grad():
            logits, _ = model(x_seq, x_stat, x_burst, seq_len, return_intermediate=True)
            pred_class = logits[0].argmax().item()
        importance = compute_gradcam_packet_importance(model, x_seq, x_stat, x_burst, seq_len, pred_class)
        real_len = seq_len[0].item()
        plt.figure(figsize=(12, 4))
        plt.bar(range(real_len), importance[:real_len], color='coral', edgecolor='darkred')
        plt.xlabel('Packet Position'); plt.ylabel('Gradient-Based Importance')
        plt.title(f'Grad-CAM Packet Importance - {class_names[pred_class]}')
        plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()


# In[16]:


# Build model with the SAME encoder size as UCDavis — required for weight loading to work
model = MultiBranchNet(
    X_stat_train.shape[1], X_burst_train.shape[1], n_classes,
    use_seq=True, use_stat=True, use_burst=True, dropout=0.2,
    seq_cnn_channels=48, seq_gru_hidden=96, seq_num_gru_layers=2,   # match UCDavis, not the 24/48/1 CPU config
)

pretrained = torch.load("ucdavis_model.pt", map_location=device)
model_dict = model.state_dict()

compatible = {k: v for k, v in pretrained.items()
              if k in model_dict and v.shape == model_dict[k].shape}
skipped = [k for k in pretrained if k not in compatible]

model_dict.update(compatible)
model.load_state_dict(model_dict)
model = model.to(device)

print(f"Loaded {len(compatible)}/{len(pretrained)} tensors from UCDavis checkpoint")
print(f"Skipped (expected: classifier.* — different n_classes): {skipped}")


# In[17]:


for name, p in model.named_parameters():
    if not name.startswith("classifier"):
        p.requires_grad = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Training {trainable:,}/{total:,} params (classifier head only)")

model, history_head, best_head = train_with_early_stopping(
    model, train_loader, val_loader,
    max_epochs=40, learning_rate=1e-3, patience=10, verbose=True,
    compute_class_weights=True, use_ema=False,
)

frozen_val_res = evaluate_model(model, val_loader, le.classes_, label="Transfer (frozen backbone)")


# In[18]:


for p in model.parameters():
    p.requires_grad = True

model, history_ft, best_ft = train_with_early_stopping(
    model, train_loader, val_loader,
    max_epochs=40, learning_rate=1e-5, patience=10, verbose=True,
    compute_class_weights=True, use_ema=False,
)

test_res = evaluate_model(model, test_loader, le.classes_, label="Transfer (fine-tuned)")

print(f"\n{'='*70}")
print(f"CROSS-DATASET GENERALIZATION SUMMARY (UCDavis -> CESNET-QUIC22)")
print(f"{'='*70}")
print(f"Frozen backbone (linear probe) val F1: {frozen_val_res['f1']:.4f}")
print(f"Fine-tuned test F1:                    {test_res['f1']:.4f}")


# In[ ]:




