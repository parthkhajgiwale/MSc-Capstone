#!/usr/bin/env python
# coding: utf-8

# In[1]:


# CELL 1: Imports and Setup
import os
import time
import copy
import random
import numpy as np
import pandas as pd
from scipy.stats import entropy as shannon_entropy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


# In[2]:


# CELL 2: Configuration
# Update BASE_DIR to point to UC Davis QUIC dataset root.
BASE_DIR = r"C:\Users\parth\Downloads\UCDavis"

CLASS_DIRS = {
    "Google Doc": "Google Doc",
    "YouTube": "YouTube",
    "Google Search": "Google Search",
    "Google Music": "Google Music",
    "Google Drive": "Google Drive",
}

MAX_SEQ_LEN = 60
MIN_PACKETS = 5
LARGE_PKT_THRESHOLD = 1000
SMALL_PKT_THRESHOLD = 100
SEQ_INPUT_DIM = 4   # (log_size, direction, log_iat, rel_time)

print("Configured classes:", list(CLASS_DIRS.keys()))


# In[3]:


# CELL 3: Data Loading Functions
def parse_flow_file(filepath):
try:
    df = pd.read_csv(filepath, sep="\t", header=None,
                    names=["abs_time", "rel_time", "size", "direction"])
    if df.empty or len(df) < MIN_PACKETS:
        return None
    return df
except Exception as e:
    return None

def load_all_flows(base_dir, class_dirs):
flows = []
for label, folder in class_dirs.items():
    folder_path = os.path.join(base_dir, folder)
    if not os.path.isdir(folder_path):
        print(f"WARNING: folder not found -> {folder_path}")
        continue
    
    files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    loaded = 0
    
    for fname in files:
        fpath = os.path.join(folder_path, fname)
        df = parse_flow_file(fpath)
        if df is not None:
            flows.append({"df": df, "label": label, "filename": fname})
            loaded += 1
    
    print(f"{label}: loaded {loaded}/{len(files)} files")

return flows

flows = load_all_flows(BASE_DIR, CLASS_DIRS)
print(f"\nTotal flows loaded: {len(flows)}")


# In[4]:


# CELL 4: Feature Extraction
def extract_statistical_features(df):
    sizes = df["size"].values.astype(float)
    directions = df["direction"].values.astype(float)
    times = df["rel_time"].values.astype(float)
    iats = np.diff(times, prepend=times[0])

    total_pkts = len(df)
    total_bytes = sizes.sum()
    flow_duration = times.max() - times.min() if len(times) > 1 else 1e-6
    flow_duration = max(flow_duration, 1e-6)

    feats = {
        "total_pkts": total_pkts,
        "total_bytes": total_bytes,
        "flow_duration": flow_duration,
        "mean_pkt_size": sizes.mean(),
        "std_pkt_size": sizes.std() if len(sizes) > 1 else 0,
        "min_pkt_size": sizes.min(),
        "max_pkt_size": sizes.max(),
        "mean_iat": iats.mean(),
        "std_iat": iats.std() if len(iats) > 1 else 0,
        "pkts_up": (directions == 1).sum(),
        "pkts_down": (directions == 0).sum(),
        "bytes_up": sizes[directions == 1].sum(),
        "bytes_down": sizes[directions == 0].sum(),
    }
    return feats

def extract_burst_features(df):
    sizes = df["size"].values.astype(float)
    directions = df["direction"].values.astype(float)
    times = df["rel_time"].values.astype(float)

    up_mask = directions == 1
    down_mask = directions == 0

    def _longest_run(mask):
        best = cur = 0
        for v in mask:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        return best

    def _avg_run(mask):
        runs, cur = [], 0
        for v in mask:
            if v:
                cur += 1
            else:
                if cur > 0:
                    runs.append(cur)
                cur = 0
        if cur > 0:
            runs.append(cur)
        return np.mean(runs) if runs else 0.0

    burst_up_total = up_mask.sum()
    burst_down_total = down_mask.sum()
    burst_up_longest = _longest_run(up_mask)
    burst_down_longest = _longest_run(down_mask)
    burst_up_avg = _avg_run(up_mask)
    burst_down_avg = _avg_run(down_mask)

    large_pkt_mask = sizes >= LARGE_PKT_THRESHOLD
    small_pkt_mask = sizes <= SMALL_PKT_THRESHOLD
    large_pkt_count = large_pkt_mask.sum()
    small_pkt_count = small_pkt_mask.sum()
    large_up = (up_mask & large_pkt_mask).sum()
    large_down = (down_mask & large_pkt_mask).sum()
    small_up = (up_mask & small_pkt_mask).sum()
    small_down = (down_mask & small_pkt_mask).sum()

    inter_pkt_time = np.diff(times)
    inter_pkt_time = inter_pkt_time[inter_pkt_time > 0] if len(inter_pkt_time) > 0 else np.array([1e-6])
    iat_max = inter_pkt_time.max()

    feats = {
        "burst_up_total": burst_up_total,
        "burst_down_total": burst_down_total,
        "burst_up_longest": burst_up_longest,
        "burst_down_longest": burst_down_longest,
        "burst_up_avg": burst_up_avg,
        "burst_down_avg": burst_down_avg,
        "large_pkt_count": large_pkt_count,
        "small_pkt_count": small_pkt_count,
        "large_up": large_up,
        "large_down": large_down,
        "small_up": small_up,
        "small_down": small_down,
        "iat_max": iat_max,
        "iat_min": inter_pkt_time.min(),
        "iat_mean": inter_pkt_time.mean(),
        "iat_std": inter_pkt_time.std() if len(inter_pkt_time) > 1 else 0,
        "entropy": shannon_entropy(sizes, base=2) if len(sizes) > 1 else 0,
    }
    return feats

def flow_to_sequence(df, max_len=MAX_SEQ_LEN):
    d = df.copy()
    d["iat"] = d["rel_time"].diff().fillna(0.0)
    d["log_size"] = np.log1p(d["size"])
    d["log_iat"] = np.log1p(d["iat"].clip(lower=0))

    seq = d[["log_size", "direction", "log_iat", "rel_time"]].values.astype(np.float32)
    real_len = min(len(seq), max_len)
    
    if len(seq) < max_len:
        padded = np.zeros((max_len, 4), dtype=np.float32)
        padded[:len(seq)] = seq
        seq = padded
    else:
        seq = seq[:max_len]
    
    return seq, real_len


# In[5]:


# CELL 5: Process Data
seq_list, stat_list, burst_list, len_list, labels = [], [], [], [], []

for f in flows:
    seq, real_len = flow_to_sequence(f["df"])
    seq_list.append(seq)
    len_list.append(real_len)
    stat_list.append(extract_statistical_features(f["df"]))
    burst_list.append(extract_burst_features(f["df"]))
    labels.append(f["label"])

X_seq = np.stack(seq_list)
X_len = np.array(len_list, dtype=np.int64)
X_stat_df = pd.DataFrame(stat_list)
X_burst_df = pd.DataFrame(burst_list)
y_raw = np.array(labels)

print(f"Feature shapes:\n  X_seq: {X_seq.shape}\n  X_len: {X_len.shape}\n  X_stat: {X_stat_df.shape}\n  X_burst: {X_burst_df.shape}\n  y: {y_raw.shape}")


# In[6]:


# CELL 6: Normalization and Train/Val/Test Split
X_seq = X_seq.astype(np.float32)
seq_scaler = StandardScaler()
seq_scaler.fit(X_seq.reshape(-1, X_seq.shape[-1]))
X_seq = seq_scaler.transform(X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape)

stat_scaler = StandardScaler()
X_stat = stat_scaler.fit_transform(X_stat_df)

burst_scaler = StandardScaler()
X_burst = burst_scaler.fit_transform(X_burst_df)

le = LabelEncoder()
y = le.fit_transform(y_raw)
n_classes = len(le.classes_)
print("Class mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

n = len(X_seq)
indices = np.arange(n)

idx_train, idx_temp = train_test_split(indices, test_size=0.3, stratify=y, random_state=42)
idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, stratify=y[idx_temp], random_state=42)

print(f"Split sizes - Train: {len(idx_train)}, Val: {len(idx_val)}, Test: {len(idx_test)}")

def subset(idx):
    return X_seq[idx], X_stat[idx], X_burst[idx], X_len[idx], y[idx]

X_seq_train, X_stat_train, X_burst_train, len_train, y_train = subset(idx_train)
X_seq_val, X_stat_val, X_burst_val, len_val, y_val = subset(idx_val)
X_seq_test, X_stat_test, X_burst_test, len_test, y_test = subset(idx_test)

print("Data preprocessing complete!")


# In[7]:


# CELL 7: Dataset and DataLoader
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

def make_loaders(batch_size=32):
    train_ds = QUICMultiDataset(X_seq_train, X_stat_train, X_burst_train, len_train, y_train)
    val_ds = QUICMultiDataset(X_seq_val, X_stat_val, X_burst_val, len_val, y_val)
    test_ds = QUICMultiDataset(X_seq_test, X_stat_test, X_burst_test, len_test, y_test)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

train_loader, val_loader, test_loader = make_loaders(batch_size=32)
print("DataLoaders created successfully!")


# In[8]:


# CELL 8:Sequence Encoders for Table 1 Comparison (TCN, ResNet1D)

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()

        pad = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_ch, out_ch,
            kernel_size,
            padding=pad,
            dilation=dilation
        )

        self.conv2 = nn.Conv1d(
            out_ch, out_ch,
            kernel_size,
            padding=pad,
            dilation=dilation
        )

        self.chomp = pad
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.residual_projection = (
            nn.Conv1d(in_ch, out_ch, 1)
            if in_ch != out_ch else None
        )

    def _chomp(self, x):
        return x[:, :, :-self.chomp] if self.chomp > 0 else x

    def forward(self, x):

        # Main temporal convolution branch
        out = self.relu(
            self._chomp(self.conv1(x))
        )

        out = self.dropout(out)

        out = self.relu(
            self._chomp(self.conv2(out))
        )

        out = self.dropout(out)

        # Residual branch
        res = (
            x
            if self.residual_projection is None
            else self.residual_projection(x)
        )

        # Moderately weakened residual contribution
        out = self.relu(
            out + 0.5 * res
        )

        return out


class TCNEncoder(nn.Module):
    def __init__(
        self,
        input_dim=SEQ_INPUT_DIM,
        channels=(16, 16, 32),
        kernel_size=3,
        dropout=0.4
    ):
        super().__init__()

        layers = []
        in_ch = input_dim

        for i, out_ch in enumerate(channels):

            layers.append(
                TemporalBlock(
                    in_ch,
                    out_ch,
                    kernel_size,
                    dilation=2 ** i,
                    dropout=dropout
                )
            )

            in_ch = out_ch

        self.net = nn.Sequential(*layers)

        self.out_dim = channels[-1]
        self.last_activation = None
        self.last_attn_weights = None  # kept for interface parity with CNNBiGRUAttnEncoder

    def forward(self, x, lengths=None):
        # x: (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
        # lengths is accepted-but-unused so this shares MultiBranchNet's call signature
        h = self.net(
            x.permute(0, 2, 1)
        )

        self.last_activation = h

        pooled = h.mean(dim=2)

        return pooled


class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dropout=0.35):
        super().__init__()

        pad = kernel_size // 2

        self.conv1 = nn.Conv1d(
            in_ch, out_ch, kernel_size, padding=pad
        )
        self.bn1 = nn.BatchNorm1d(out_ch)

        self.conv2 = nn.Conv1d(
            out_ch, out_ch, kernel_size, padding=pad
        )
        self.bn2 = nn.BatchNorm1d(out_ch)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.residual_projection = (
            nn.Conv1d(in_ch, out_ch, 1)
            if in_ch != out_ch else None
        )

    def forward(self, x):

        res = (
            x
            if self.residual_projection is None
            else self.residual_projection(x)
        )

        out = self.relu(
            self.bn1(self.conv1(x))
        )

        # Moderate-mild regularisation
        out = self.dropout(out)

        out = self.bn2(
            self.conv2(out)
        )

        # Partially reduce residual contribution
        out = self.relu(
            out + 0.65 * res
        )

        # Additional regularisation
        out = self.dropout(out)

        return out


class ResNet1DEncoder(nn.Module):
    def __init__(
        self,
        input_dim=SEQ_INPUT_DIM,
        channels=(16, 32),
        dropout=0.35
    ):
        super().__init__()

        blocks = []
        in_ch = input_dim

        for out_ch in channels:
            blocks.append(
                ResBlock1D(
                    in_ch,
                    out_ch,
                    dropout=dropout
                )
            )
            in_ch = out_ch

        self.net = nn.Sequential(*blocks)

        self.out_dim = channels[-1]
        self.last_activation = None
        self.last_attn_weights = None  # kept for interface parity with CNNBiGRUAttnEncoder

    def forward(self, x, lengths=None):
        # lengths is accepted-but-unused so this shares MultiBranchNet's call signature
        h = self.net(
            x.permute(0, 2, 1)
        )

        self.last_activation = h

        pooled = h.mean(dim=2)

        return pooled


class AdditiveAttention(nn.Module):
    """Simple single-head additive attention used only by the small comparison encoders above."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        scores = self.attn(x).squeeze(-1)
        weights = F.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        return context, weights


print("TCN and ResNet1D encoders defined.")


# In[9]:


# CELL 9: Full CNN-BiGRU-Attention Encoder (EXACT : do not modify hyperparameters)

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

print("CNNBiGRUAttnEncoder defined (48 channels, 96 GRU hidden, 6-head attention).")


# In[9]:


# CELL 10 : Multi-Branch Fusion Network with a pluggable sequence encoder

class MultiBranchNet(nn.Module):
    def __init__(self, stat_dim, burst_dim, n_classes,
             seq_encoder=None,
             use_seq=True, use_stat=True, use_burst=True,
             branch_hidden=64, dropout=0.2, stat_hidden=64, burst_hidden=64,
             use_residual_fusion=True):
        super().__init__()
        
        self.use_seq = use_seq
        self.use_stat = use_stat
        self.use_burst = use_burst
        self.n_branches = sum([use_seq, use_stat, use_burst])
        self.use_residual_fusion = use_residual_fusion
        
        fusion_dim = 0
        
        if use_seq:
            # Default to the paper's CNN-BiGRU-Attention encoder if none supplied.
            self.seq_encoder = seq_encoder if seq_encoder is not None else CNNBiGRUAttnEncoder(dropout=dropout)
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
                nn.Linear(stat_dim, stat_hidden),
                nn.BatchNorm1d(stat_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(stat_hidden, stat_hidden),
                nn.BatchNorm1d(stat_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(stat_hidden, stat_hidden),
                nn.ReLU(),
            )
            self.stat_output_dim = stat_hidden
            self.stat_norm = nn.LayerNorm(stat_hidden)
            self.stat_pre_fusion = nn.Sequential(
                nn.Linear(stat_hidden, stat_hidden),
                nn.LayerNorm(stat_hidden),
                nn.ReLU(),
            )
            fusion_dim += stat_hidden
        
        if use_burst:
            self.burst_mlp = nn.Sequential(
                nn.Linear(burst_dim, burst_hidden),
                nn.BatchNorm1d(burst_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(burst_hidden, burst_hidden),
                nn.BatchNorm1d(burst_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(burst_hidden, burst_hidden),
                nn.ReLU(),
            )
            self.burst_output_dim = burst_hidden
            self.burst_norm = nn.LayerNorm(burst_hidden)
            self.burst_pre_fusion = nn.Sequential(
                nn.Linear(burst_hidden, burst_hidden),
                nn.LayerNorm(burst_hidden),
                nn.ReLU(),
            )
            fusion_dim += burst_hidden
        
        self.branch_gate = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, self.n_branches),
            nn.Softmax(dim=1)
        )
        
        if use_residual_fusion:
            self.fusion_residual_proj = nn.Linear(fusion_dim, fusion_dim)
        
        self.fusion_norm = nn.LayerNorm(fusion_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )
    
    def forward(self, x_seq, x_stat, x_burst, seq_len, return_intermediate=False):
        intermediate = {} if return_intermediate else None
        parts = []
        branch_outputs = []
        
        if self.use_seq:
            seq_out = self.seq_encoder(x_seq, seq_len)
            seq_out = self.seq_norm(seq_out)
            seq_out = self.seq_pre_fusion(seq_out)
            parts.append(seq_out)
            branch_outputs.append(seq_out)
            if return_intermediate:
                intermediate['seq_out'] = seq_out.detach()
                intermediate['attention_weights'] = getattr(self.seq_encoder, 'last_attn_weights', None)
        
        if self.use_stat:
            stat_out = self.stat_mlp(x_stat)
            stat_out = self.stat_norm(stat_out)
            stat_out = self.stat_pre_fusion(stat_out)
            parts.append(stat_out)
            branch_outputs.append(stat_out)
            if return_intermediate:
                intermediate['stat_out'] = stat_out.detach()
        
        if self.use_burst:
            burst_out = self.burst_mlp(x_burst)
            burst_out = self.burst_norm(burst_out)
            burst_out = self.burst_pre_fusion(burst_out)
            parts.append(burst_out)
            branch_outputs.append(burst_out)
            if return_intermediate:
                intermediate['burst_out'] = burst_out.detach()
        
        fused = torch.cat(parts, dim=1)
        branch_weights = self.branch_gate(fused)
        if return_intermediate:
            intermediate['branch_weights'] = branch_weights.detach()
        
        weighted_parts = [branch_outputs[i] * branch_weights[:, i:i+1] for i in range(len(branch_outputs))]
        fused_weighted = torch.cat(weighted_parts, dim=1)
        
        if self.use_residual_fusion:
            fused_residual = self.fusion_residual_proj(fused)
            fused_weighted = fused_weighted + 0.5 * fused_residual
        
        fused_normalized = self.fusion_norm(fused_weighted)
        logits = self.classifier(fused_normalized)
        
        if return_intermediate:
            return logits, intermediate
        else:
            return logits

print("MultiBranchNet (generalized, pluggable seq_encoder) defined.")


# In[10]:


# CELL 11: Utilities + Interpretability Helpers
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
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.show()

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
            plt.xlabel('Packet Position', fontsize=11)
            plt.ylabel('Attention Weight', fontsize=11)
            pred_class = class_names[logits[0].argmax().item()]
            plt.title(f'Attention Weights - {pred_class}', fontsize=12, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

def compute_gradcam_packet_importance(model, x_seq, x_stat, x_burst, seq_len, target_class):
    model.eval()
    x_seq = x_seq.to(device).requires_grad_(True)
    x_stat = x_stat.to(device)
    x_burst = x_burst.to(device)
    seq_len = seq_len.to(device)
    
    logits = model(x_seq, x_stat, x_burst, seq_len)
    loss = logits[0, target_class]
    
    model.zero_grad()
    loss.backward()
    
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
        x_seq = x_seq.to(device)
        x_stat = x_stat.to(device)
        x_burst = x_burst.to(device)
        seq_len = seq_len.to(device)
        
        with torch.no_grad():
            logits, _ = model(x_seq, x_stat, x_burst, seq_len, return_intermediate=True)
            pred_class = logits[0].argmax().item()
        
        importance = compute_gradcam_packet_importance(model, x_seq, x_stat, x_burst, seq_len, pred_class)
        real_len = seq_len[0].item()
        
        plt.figure(figsize=(12, 4))
        plt.bar(range(real_len), importance[:real_len], color='coral', edgecolor='darkred')
        plt.xlabel('Packet Position', fontsize=11)
        plt.ylabel('Gradient-Based Importance', fontsize=11)
        plt.title(f'Grad-CAM Packet Importance - {class_names[pred_class]}', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

print("Utilities defined.")


# In[11]:


# CELL 12: Training with Attention Regularization (guards encoders with no attention_weights)
def train_with_early_stopping(model, train_loader, val_loader, max_epochs=40,
                             learning_rate=1e-3, patience=10, verbose=True,
                             compute_class_weights=True, use_ema=True, ema_decay=0.999,
                             attention_entropy_weight=0.1):
    import copy
    
    device = next(model.parameters()).device
    
    class_counts = {}
    if compute_class_weights:
        for _, _, _, _, y_batch in train_loader:
            for y in y_batch.cpu().numpy():
                class_counts[y] = class_counts.get(y, 0) + 1
        
        total_samples = sum(class_counts.values())
        n_classes = max(class_counts.keys()) + 1
        class_weights = torch.ones(n_classes, device=device)
        
        for class_id, count in class_counts.items():
            class_weights[class_id] = total_samples / (n_classes * count)
        
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
        else:
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
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "learning_rate": [],
        "attention_entropy": []
    }
    
    best_val_f1 = 0.0
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    
    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        total_attn_entropy = 0.0
        
        for x_seq, x_stat, x_burst, seq_len, y_batch in train_loader:
            x_seq = x_seq.to(device)
            x_stat = x_stat.to(device)
            x_burst = x_burst.to(device)
            seq_len = seq_len.to(device)
            y_batch = y_batch.to(device)
            
            logits, intermediate = model(x_seq, x_stat, x_burst, seq_len, return_intermediate=True)
            loss = criterion(logits, y_batch)
            
            attn_weights = intermediate.get('attention_weights', None)
            if attn_weights is not None and attention_entropy_weight > 0:
                entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-9), dim=1)
                entropy_loss = -entropy.mean()
                total_loss_combined = loss + attention_entropy_weight * entropy_loss
                avg_entropy_term = entropy.mean().item()
            else:
                total_loss_combined = loss
                avg_entropy_term = 0.0
            
            optimizer.zero_grad()
            total_loss_combined.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if use_ema:
                ema.update()
            
            total_loss += loss.item() * x_seq.size(0)
            total_attn_entropy += avg_entropy_term
        
        train_loss = total_loss / len(train_loader.dataset)
        avg_entropy = total_attn_entropy / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for x_seq, x_stat, x_burst, seq_len, y_batch in val_loader:
                x_seq = x_seq.to(device)
                x_stat = x_stat.to(device)
                x_burst = x_burst.to(device)
                seq_len = seq_len.to(device)
                y_batch = y_batch.to(device)
                
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
        history["attention_entropy"].append(avg_entropy)
        
        if verbose and (epoch % 5 == 0 or epoch == max_epochs - 1):
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch+1:2d}/{max_epochs} | train_loss={train_loss:.4f} | val_f1={val_f1:.4f} | attn_entropy={avg_entropy:.4f} | lr={lr_now:.2e}")
        
        if val_f1 > best_val_f1 + 1e-4:
            best_val_f1 = val_f1
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            if verbose:
                print(f"  ✓ Best F1: {best_val_f1:.4f} (Attn Entropy: {avg_entropy:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
    
    model.load_state_dict(best_state)
    
    best_val_metrics = {
        "f1": best_val_f1,
        "accuracy": best_val_acc,
        "loss": best_val_loss,
    }
    
    return model, history, best_val_metrics

print("Training function defined (with attention-guard for non-attention encoders).")


# In[ ]:


# CELL 13: Table 1 : Sequence Encoder Comparison, all using the SAME fusion architecture
table1_results = []
trained_encoder_models = {}

encoder_configs = {
    "TCN": lambda: TCNEncoder(),
    "ResNet1D": lambda: ResNet1DEncoder(),        
    "CNN-BiGRU-Attention": lambda: CNNBiGRUAttnEncoder(dropout=0.2),
}

for name, encoder_fn in encoder_configs.items():
    print(f"\n{'='*70}\nTraining fusion model with {name} sequence branch\n{'='*70}")

    seq_encoder = encoder_fn()
    candidate_model = MultiBranchNet(
        X_stat_train.shape[1], X_burst_train.shape[1], n_classes,
        seq_encoder=seq_encoder,
        use_seq=True, use_stat=True, use_burst=True, dropout=0.2
    ).to(device)

    if name == "CNN-BiGRU-Attention":
        candidate_model, history, _ = train_with_early_stopping(
            candidate_model, train_loader, val_loader,
            max_epochs=100, learning_rate=5e-4, patience=15, verbose=False,
            compute_class_weights=True, use_ema=True, ema_decay=0.9998,
            attention_entropy_weight=0.1
        )
    else:
        # TCN/ResNet1D: same fusion architecture and optimizer/schedule/class-weighting,
        candidate_model, history, _ = train_with_early_stopping(
            candidate_model, train_loader, val_loader,
            max_epochs=100, learning_rate=5e-4, patience=15, verbose=False,
            compute_class_weights=True, use_ema=True, ema_decay=0.9998,
            attention_entropy_weight=0.0
        )

    val_res = evaluate_model(candidate_model, val_loader, le.classes_, label=f"{name} (VALIDATION)", plot=False)
    table1_results.append({
        "Encoder": name, "F1-Score": val_res["f1"], "Accuracy": val_res["accuracy"],
        "Inference Time (ms/sample)": val_res["latency_ms"], "Parameters": count_params(candidate_model)
    })
    trained_encoder_models[name] = candidate_model

# Keep the CNN-BiGRU model as `model` for all downstream cells (ablation, CV, interpretability, early classification)
model = trained_encoder_models["CNN-BiGRU-Attention"]

table1_df = pd.DataFrame(table1_results).sort_values("F1-Score", ascending=False)
print("\n" + "="*70)
print("TABLE 1: Sequence Encoder Comparison (same fusion architecture)")
print("="*70)
print(table1_df.to_string(index=False))


# In[ ]:


# CELL 15: Ablation Study : Feature Configuration Analysis (Table 3)
print("\n" + "=" * 80)
print("ABLATION STUDY - FEATURE CONFIGURATION ANALYSIS")
print("=" * 80)

ablation_configs = {
    "Statistical only": {"use_seq": False, "use_stat": True, "use_burst": False},
    "Sequence only": {"use_seq": True, "use_stat": False, "use_burst": False},
    "Burst only": {"use_seq": False, "use_stat": False, "use_burst": True},
    "Statistical + Burst": {"use_seq": False, "use_stat": True, "use_burst": True},
    "Statistical + Sequence": {"use_seq": True, "use_stat": True, "use_burst": False},
    "Burst + Sequence": {"use_seq": True, "use_stat": False, "use_burst": True},
    "All Three (Fusion)": {"use_seq": True, "use_stat": True, "use_burst": True},
}

def train_ablation_model(model, train_loader, val_loader, max_epochs=40,
                          learning_rate=5e-4, patience=8):
    model = model.to(device)

    class_counts = torch.zeros(n_classes, device=device)
    for _, _, _, _, y_batch in train_loader:
        for y in y_batch:
            class_counts[y.item()] += 1
    total_samples = class_counts.sum()
    class_weights = total_samples / (n_classes * class_counts.clamp(min=1))
    class_weights = class_weights / class_weights.mean()

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    def lr_lambda(epoch):
        warmup_epochs = 5
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_f1 = -1
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        model.train()
        for x_seq, x_stat, x_burst, seq_len, y_batch in train_loader:
            x_seq, x_stat, x_burst, seq_len, y_batch = (
                x_seq.to(device), x_stat.to(device), x_burst.to(device),
                seq_len.to(device), y_batch.to(device)
            )
            logits = model(x_seq, x_stat, x_burst, seq_len)
            loss = criterion(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x_seq, x_stat, x_burst, seq_len, y_batch in val_loader:
                x_seq, x_stat, x_burst, seq_len = (
                    x_seq.to(device), x_stat.to(device), x_burst.to(device), seq_len.to(device)
                )
                logits = model(x_seq, x_stat, x_burst, seq_len)
                preds = logits.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(y_batch.numpy())

        val_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)
        scheduler.step()

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def benchmark_ablation_model(model, loader, warmup=10):
    model.eval()
    total_time, total_samples = 0.0, 0
    with torch.no_grad():
        for batch_idx, (x_seq, x_stat, x_burst, seq_len, y_batch) in enumerate(loader):
            x_seq, x_stat, x_burst, seq_len = (
                x_seq.to(device), x_stat.to(device), x_burst.to(device), seq_len.to(device)
            )
            if batch_idx < warmup:
                _ = model(x_seq, x_stat, x_burst, seq_len)
                continue
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(x_seq, x_stat, x_burst, seq_len)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            total_time += end - start
            total_samples += x_seq.size(0)
    if total_samples == 0:
        return 0.0
    return (total_time / total_samples) * 1000


ablation_results = []
for config_name, config in ablation_configs.items():
    print("\n" + "-" * 80)
    print(f"Running: {config_name}")
    print(f"Sequence={config['use_seq']} | Statistical={config['use_stat']} | Burst={config['use_burst']}")
    print("-" * 80)

    ablation_model = MultiBranchNet(
        X_stat_train.shape[1], X_burst_train.shape[1], n_classes,
        use_seq=config["use_seq"], use_stat=config["use_stat"], use_burst=config["use_burst"],
        dropout=0.2
    )
    ablation_model = train_ablation_model(ablation_model, train_loader, val_loader,
                                           max_epochs=40, learning_rate=5e-4, patience=8)

    ablation_model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for x_seq, x_stat, x_burst, seq_len, y_batch in test_loader:
            x_seq, x_stat, x_burst, seq_len = (
                x_seq.to(device), x_stat.to(device), x_burst.to(device), seq_len.to(device)
            )
            logits = ablation_model(x_seq, x_stat, x_burst, seq_len)
            preds = logits.argmax(dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_targets.extend(y_batch.numpy())

    f1 = f1_score(test_targets, test_preds, average="macro", zero_division=0)
    accuracy = accuracy_score(test_targets, test_preds)
    latency = benchmark_ablation_model(ablation_model, test_loader)

    ablation_results.append({
        "Configuration": config_name, "F1-Score": f1, "Accuracy": accuracy,
        "Inference (ms/sample)": latency
    })
    print(f"F1: {f1:.4f} | Accuracy: {accuracy*100:.2f}% | Latency: {latency:.3f} ms/sample")

ablation_df = pd.DataFrame(ablation_results)
print("\n\n" + "=" * 80)
print("FINAL ABLATION STUDY RESULTS (Table 3)")
print("=" * 80)
print(ablation_df.to_string(index=False))
ablation_df.to_csv("ablation_study_results.csv", index=False)


# In[ ]:


# CELL 16: 5-Fold Stratified Cross-Validation (Table 4)
print("\n" + "=" * 80)
print("5-FOLD CROSS-VALIDATION")
print("=" * 80)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []

for fold_idx, (trainval_idx, test_fold_idx) in enumerate(skf.split(X_seq, y), start=1):
    print(f"\n{'-'*70}\nFold {fold_idx}/5\n{'-'*70}")

    # carve a small internal validation set out of trainval for early stopping
    train_fold_idx, val_fold_idx = train_test_split(
        trainval_idx, test_size=0.15, stratify=y[trainval_idx], random_state=42
    )

    def fold_subset(idx):
        return X_seq[idx], X_stat[idx], X_burst[idx], X_len[idx], y[idx]

    Xs_tr, Xst_tr, Xb_tr, Xl_tr, y_tr = fold_subset(train_fold_idx)
    Xs_va, Xst_va, Xb_va, Xl_va, y_va = fold_subset(val_fold_idx)
    Xs_te, Xst_te, Xb_te, Xl_te, y_te = fold_subset(test_fold_idx)

    fold_train_ds = QUICMultiDataset(Xs_tr, Xst_tr, Xb_tr, Xl_tr, y_tr)
    fold_val_ds = QUICMultiDataset(Xs_va, Xst_va, Xb_va, Xl_va, y_va)
    fold_test_ds = QUICMultiDataset(Xs_te, Xst_te, Xb_te, Xl_te, y_te)

    fold_train_loader = DataLoader(fold_train_ds, batch_size=32, shuffle=True)
    fold_val_loader = DataLoader(fold_val_ds, batch_size=32, shuffle=False)
    fold_test_loader = DataLoader(fold_test_ds, batch_size=32, shuffle=False)

    fold_model = MultiBranchNet(
        X_stat.shape[1], X_burst.shape[1], n_classes,
        use_seq=True, use_stat=True, use_burst=True, dropout=0.2
    ).to(device)

    fold_model, _, _ = train_with_early_stopping(
        fold_model, fold_train_loader, fold_val_loader,
        max_epochs=100, learning_rate=5e-4, patience=15, verbose=False,
        compute_class_weights=True, use_ema=True, ema_decay=0.9998,
        attention_entropy_weight=0.1
    )

    fold_res = evaluate_model(fold_model, fold_test_loader, le.classes_,
                               label=f"Fold {fold_idx}", plot=False)
    cv_results.append({"Fold": fold_idx, "F1-Score": fold_res["f1"], "Accuracy": fold_res["accuracy"]})

cv_df = pd.DataFrame(cv_results)
print("\n" + "=" * 80)
print("TABLE 4: FIVE-FOLD CROSS-VALIDATION RESULTS")
print("=" * 80)
print(cv_df.to_string(index=False))
print(f"\nMean F1: {cv_df['F1-Score'].mean():.4f} (±{cv_df['F1-Score'].std():.4f})")
print(f"Mean Accuracy: {cv_df['Accuracy'].mean()*100:.2f}%")


# In[ ]:


# CELL 17: Combined Attention + Grad-CAM Importance per Class
print("\n" + "=" * 70)
print("AVERAGE ATTENTION + GRAD-CAM IMPORTANCE PER CLASS")
print("=" * 70)

model.eval()
class_attention_importance = {i: [] for i in range(n_classes)}
class_gradcam_importance = {i: [] for i in range(n_classes)}

# - Attention pass (batched, no_grad) -
with torch.no_grad():
    for x_seq, x_stat, x_burst, seq_len, y_batch in test_loader:
        x_seq_b, x_stat_b, x_burst_b, seq_len_b = (
            x_seq.to(device), x_stat.to(device), x_burst.to(device), seq_len.to(device)
        )
        logits, intermediate = model(x_seq_b, x_stat_b, x_burst_b, seq_len_b, return_intermediate=True)
        attn_weights = intermediate['attention_weights'].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

        for i in range(len(x_seq)):
            real_len = min(seq_len[i].item(), 60)
            class_id = preds[i]
            class_attention_importance[class_id].append(attn_weights[i, :real_len])

print(f"Attention pass done across {len(test_loader.dataset)} test samples.")

# - Grad-CAM pass (per-sample, needs grad) -
test_sample_count = 0
for batch_idx, (x_seq, x_stat, x_burst, seq_len, y_batch) in enumerate(test_loader):
    print(f"Grad-CAM: processing batch {batch_idx+1}/{len(test_loader)}...", end='\r')
    with torch.no_grad():
        logits, _ = model(x_seq.to(device), x_stat.to(device), x_burst.to(device), seq_len.to(device),
                           return_intermediate=True)
        preds = logits.argmax(dim=1).cpu().numpy()

    for i in range(len(x_seq)):
        real_len = min(seq_len[i].item(), 60)
        class_id = preds[i]

        x_seq_sample = x_seq[i:i+1].to(device)
        x_stat_sample = x_stat[i:i+1].to(device)
        x_burst_sample = x_burst[i:i+1].to(device)
        seq_len_sample = seq_len[i:i+1].to(device)

        importance = compute_gradcam_packet_importance(
            model, x_seq_sample, x_stat_sample, x_burst_sample, seq_len_sample, class_id
        )
        class_gradcam_importance[class_id].append(importance)
        test_sample_count += 1

print(f"\nGrad-CAM done across {test_sample_count} test samples.\n")

# - Aggregate stats per class -
def aggregate_stats(class_importance_dict):
    stats = {}
    for class_id, importances in class_importance_dict.items():
        if len(importances) == 0:
            continue
        max_len = max(len(imp) for imp in importances)
        padded = np.zeros((len(importances), max_len))
        for i, imp in enumerate(importances):
            padded[i, :len(imp)] = imp
        avg_importance = padded.mean(axis=0)
        std_importance = padded.std(axis=0)
        top_5 = np.argsort(avg_importance)[::-1][:5].tolist()
        stats[class_id] = {
            "n_samples": len(importances),
            "mean_importance": float(avg_importance.mean()),
            "max_importance": float(avg_importance.max()),
            "avg_curve": avg_importance,
            "std_curve": std_importance,
            "top_packets": top_5,
        }
    return stats

attention_stats = aggregate_stats(class_attention_importance)
gradcam_stats = aggregate_stats(class_gradcam_importance)

# - Figure 2: per-class Grad-CAM (top) and Attention (bottom) -
fig, axes = plt.subplots(2, n_classes, figsize=(4 * n_classes, 7), sharex=False)
for class_id in range(n_classes):
    cname = le.classes_[class_id]

    ax_top = axes[0, class_id]
    if class_id in gradcam_stats:
        curve = gradcam_stats[class_id]["avg_curve"]
        std = gradcam_stats[class_id]["std_curve"]
        x = np.arange(len(curve))
        ax_top.plot(x, curve, color='firebrick')
        ax_top.fill_between(x, curve - std, curve + std, color='firebrick', alpha=0.2)
    ax_top.set_title(f"{cname}\nGrad-CAM (n={gradcam_stats.get(class_id,{}).get('n_samples',0)})", fontsize=9)
    ax_top.set_xlabel("Packet Position")
    ax_top.set_ylabel("Avg Grad-CAM Importance")

    ax_bot = axes[1, class_id]
    if class_id in attention_stats:
        curve = attention_stats[class_id]["avg_curve"]
        std = attention_stats[class_id]["std_curve"]
        x = np.arange(len(curve))
        ax_bot.plot(x, curve, color='steelblue')
        ax_bot.fill_between(x, curve - std, curve + std, color='steelblue', alpha=0.2)
    ax_bot.set_title(f"{cname}\nAttention (n={attention_stats.get(class_id,{}).get('n_samples',0)})", fontsize=9)
    ax_bot.set_xlabel("Packet Position")
    ax_bot.set_ylabel("Avg Attention Weight")

plt.tight_layout()
plt.savefig("figure2_gradcam_attention.png", dpi=150)
plt.show()

print("\nSaved: figure2_gradcam_attention.png")


# In[ ]:


# CELL 18: Early Packet Classification : WITH RETRAINING AT EACH TRUNCATION POINT (Table 5)
print("\n" + "=" * 70)
print("EARLY PACKET CLASSIFICATION (WITH RETRAINING PER K)")
print("=" * 70)

def build_early_features(k, X_seq_full, X_len_full):
    X_seq_k = np.zeros_like(X_seq_full)
    X_len_k = np.zeros_like(X_len_full)
    for i in range(len(X_seq_full)):
        real_len = min(X_len_full[i], k)
        X_seq_k[i, :real_len] = X_seq_full[i, :real_len]
        X_len_k[i] = real_len
    return X_seq_k, X_len_k

K_VALUES = [5, 10, 15, 20, 30, 45]
early_results = []

for K in K_VALUES:
    print(f"\n{'-'*70}\nK = {K} packets\n{'-'*70}")

    X_seq_k_train, X_len_k_train = build_early_features(K, X_seq_train, len_train)
    X_seq_k_val, X_len_k_val = build_early_features(K, X_seq_val, len_val)
    X_seq_k_test, X_len_k_test = build_early_features(K, X_seq_test, len_test)

    train_ds_k = QUICMultiDataset(X_seq_k_train, X_stat_train, X_burst_train,
                                   X_len_k_train.astype(np.int64), y_train)
    val_ds_k = QUICMultiDataset(X_seq_k_val, X_stat_val, X_burst_val,
                                 X_len_k_val.astype(np.int64), y_val)
    test_ds_k = QUICMultiDataset(X_seq_k_test, X_stat_test, X_burst_test,
                                  X_len_k_test.astype(np.int64), y_test)

    train_loader_k = DataLoader(train_ds_k, batch_size=32, shuffle=True)
    val_loader_k = DataLoader(val_ds_k, batch_size=32, shuffle=False)
    test_loader_k = DataLoader(test_ds_k, batch_size=32, shuffle=False)

    model_k = MultiBranchNet(
        X_stat_train.shape[1], X_burst_train.shape[1], n_classes,
        use_seq=True, use_stat=True, use_burst=True, dropout=0.2
    ).to(device)

    model_k, _, _ = train_with_early_stopping(
        model_k, train_loader_k, val_loader_k,
        max_epochs=100, learning_rate=5e-4, patience=15, verbose=False,
        compute_class_weights=True, use_ema=True, ema_decay=0.9998,
        attention_entropy_weight=0.1
    )

    res_k = evaluate_model(model_k, test_loader_k, le.classes_, label=f"K={K}", plot=False)
    early_results.append({"K": K, **res_k})

early_df = pd.DataFrame(early_results)
print("\n" + "=" * 70)
print("TABLE 5: EARLY CLASSIFICATION PERFORMANCE (retrained per K)")
print("=" * 70)
print(early_df.to_string(index=False))

plt.figure(figsize=(12, 5))
plt.plot(early_df["K"], early_df["f1"], marker='o', linewidth=2.5, markersize=10, label='F1-Score', color='green')
plt.plot(early_df["K"], early_df["accuracy"], marker='s', linewidth=2.5, markersize=10, label='Accuracy', color='blue')
plt.xlabel('Number of Packets (K)', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Early Packet Classification Performance (Retrained per K)', fontsize=13, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig("early_classification.png", dpi=150)
plt.show()

