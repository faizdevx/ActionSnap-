"""
ActionClassificationTransformer
=================================
Drop-in replacement for src/lstm.py (ActionClassificationLSTM).
All public names that app.py and the training notebook depend on are kept
identical so the rest of the repo needs only a one-line import change.

Architecture
------------
Input  : (B, 32, 36)   — 32 frames, 36 features (18 keypoints × x/y)
Patch  : Linear 36 → 128
PosEnc : Sinusoidal (learnable-free, works for any seq len)
Encoder: 4 × TransformerEncoderLayer (d=128, heads=8, ffn=256, drop=0.1)
Pool   : Global average pooling over the time axis
Head   : Linear 128 → num_classes (6)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

# ──────────────────────────────────────────────────────────────────────────────
# Dataset / DataModule  (identical interface to PoseDataModule in lstm.py)
# ──────────────────────────────────────────────────────────────────────────────

LABELS = [
    "JUMPING",
    "JUMPING_JACKS",
    "BOXING",
    "WAVING_2HANDS",
    "WAVING_1HAND",
    "CLAPPING_HANDS",
]

N_STEPS   = 32   # frames per clip
N_FEATURES = 36  # 18 keypoints × 2 (x, y)


def _load_X(path: str) -> np.ndarray:
    with open(path) as f:
        rows = [row.split(",") for row in f]
    arr   = np.array(rows, dtype=np.float32)
    clips = len(arr) // N_STEPS
    return np.array(np.split(arr, clips))  # (N, 32, 36)


def _load_y(path: str) -> np.ndarray:
    with open(path) as f:
        rows = [row.replace("  ", " ").strip().split(" ") for row in f]
    return np.array(rows, dtype=np.int32) - 1  # 0-indexed, shape (N, 1)


class PoseDataModule(pl.LightningDataModule):
    """Loads the RNN-HAR-2D-Pose-database and exposes train / val loaders."""

    def __init__(self, data_root: str, batch_size: int = 512):
        super().__init__()
        self.data_root  = data_root
        self.batch_size = batch_size

    def setup(self, stage=None):
        X_tr = _load_X(f"{self.data_root}X_train.txt")
        y_tr = _load_y(f"{self.data_root}Y_train.txt").squeeze()
        X_te = _load_X(f"{self.data_root}X_test.txt")
        y_te = _load_y(f"{self.data_root}Y_test.txt").squeeze()

        self.train_ds = TensorDataset(
            torch.tensor(X_tr), torch.tensor(y_tr, dtype=torch.long)
        )
        self.val_ds = TensorDataset(
            torch.tensor(X_te), torch.tensor(y_te, dtype=torch.long)
        )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,  num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_ds,   batch_size=self.batch_size, shuffle=False, num_workers=2)


# ──────────────────────────────────────────────────────────────────────────────
# Sinusoidal Positional Encoding
# ──────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Standard fixed sin/cos encoding — no learnable params needed."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)                       # (L, d)
        pos = torch.arange(max_len).unsqueeze(1).float()         # (L, 1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)                                      # (1, L, d)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, T, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ──────────────────────────────────────────────────────────────────────────────
# Transformer Classifier  (public name kept == ActionClassificationLSTM
#                           so app.py import changes only the module path)
# ──────────────────────────────────────────────────────────────────────────────

class ActionClassificationTransformer(pl.LightningModule):
    """
    Spatiotemporal Transformer for human-action classification.

    Parameters
    ----------
    input_size   : int   — feature dim per frame (36 for 18 keypoints)
    d_model      : int   — internal representation dim (128)
    nhead        : int   — attention heads (8)
    num_layers   : int   — encoder depth (4)
    dim_feedforward: int — FFN hidden size (256)
    dropout      : float — applied after attention & pos-enc (0.1)
    num_classes  : int   — output classes (6)
    learning_rate: float — Adam lr (1e-4)
    """

    def __init__(
        self,
        input_size:      int   = N_FEATURES,
        d_model:         int   = 128,
        nhead:           int   = 8,
        num_layers:      int   = 4,
        dim_feedforward: int   = 256,
        dropout:         float = 0.1,
        num_classes:     int   = 6,
        learning_rate:   float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        # 1. Patch / frame embedding  (linear projection per frame)
        self.input_proj = nn.Linear(input_size, d_model)

        # 2. Positional encoding
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        # 3. Transformer encoder stack
        enc_layer = nn.TransformerEncoderLayer(
            d_model        = d_model,
            nhead          = nhead,
            dim_feedforward= dim_feedforward,
            dropout        = dropout,
            batch_first    = True,   # (B, T, d) convention
            norm_first     = True,   # Pre-LN for stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            enc_layer, num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        # 4. Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

        self.criterion = nn.CrossEntropyLoss()

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T, input_size)  e.g. (B, 32, 36)
        returns logits (B, num_classes)
        """
        x = self.input_proj(x)           # (B, T, d_model)
        x = self.pos_enc(x)              # add temporal info
        x = self.transformer_encoder(x)  # (B, T, d_model)
        x = x.mean(dim=1)               # global average pool over time
        return self.classifier(x)        # (B, num_classes)

    # ── Lightning steps ───────────────────────────────────────────────────────

    def _shared_step(self, batch, stage: str):
        X, y = batch
        logits = self(X)
        loss   = self.criterion(logits, y)
        acc    = (logits.argmax(dim=1) == y).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc",  acc,  prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }

    # ── Inference helper (mirrors lstm.py interface used by video_analyzer) ───

    @torch.no_grad()
    def predict_label(self, pose_sequence: np.ndarray) -> str:
        """
        pose_sequence : ndarray (32, 36) — one clip
        returns       : str label
        """
        self.eval()
        x      = torch.tensor(pose_sequence, dtype=torch.float32).unsqueeze(0)
        logits = self(x)
        idx    = logits.argmax(dim=1).item()
        return LABELS[idx]


# ── Backward-compat alias so a simple sed/import change is all that's needed ──
# app.py does:  from src.lstm import ActionClassificationLSTM
# After change: from src.transformer import ActionClassificationLSTM
ActionClassificationLSTM = ActionClassificationTransformer
