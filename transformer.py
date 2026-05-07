import os
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

# ─────────────────────────────────────────────────────────────
# Constants & Labels
# ─────────────────────────────────────────────────────────────

LABELS = [
    "JUMPING",
    "JUMPING_JACKS",
    "BOXING",
    "WAVING_2HANDS",
    "WAVING_1HAND",
    "CLAPPING_HANDS",
]

N_STEPS = 32   # frames per clip
N_FEATURES = 36  # 18 keypoints × 2 (x, y)
REQUIRED_DATA_FILES = ("X_train.txt", "Y_train.txt", "X_test.txt", "Y_test.txt")

# ─────────────────────────────────────────────────────────────
# Dataset Loading Helpers
# ─────────────────────────────────────────────────────────────

def _resolve_data_root(data_root: str) -> Path:
    root = Path(data_root).expanduser()
    candidates = [
        root,
        Path.cwd() / root,
        Path.cwd() / "content" / root.name,
        Path.cwd() / root.name,
    ]

    if root.is_absolute():
        candidates.append(Path.cwd() / Path(*root.parts[1:]))

    for candidate in candidates:
        if all((candidate / filename).exists() for filename in REQUIRED_DATA_FILES):
            return candidate

    checked = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Could not find the pose dataset files. Checked:\n"
        f"{checked}\n"
        "Expected X_train.txt, Y_train.txt, X_test.txt, and Y_test.txt."
    )


def _load_X(path: str | Path) -> np.ndarray:
    with open(path) as f:
        rows = [row.split(",") for row in f]
    arr = np.array(rows, dtype=np.float32)
    clips = len(arr) // N_STEPS
    return np.array(np.split(arr, clips))  # (N, 32, 36)

def _load_y(path: str | Path) -> np.ndarray:
    with open(path) as f:
        # Handle potential double spaces in source text files
        rows = [row.replace("  ", " ").strip().split(" ") for row in f]
    return np.array(rows, dtype=np.int32) - 1  # 0-indexed


# ─────────────────────────────────────────────────────────────
# DataModule
# ─────────────────────────────────────────────────────────────

class PoseDataModule(pl.LightningDataModule):
    def __init__(self, data_root: str, batch_size: int = 512, num_workers: int = 0):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        data_root = _resolve_data_root(self.data_root)
        self.data_root = str(data_root)

        X_tr = _load_X(data_root / "X_train.txt")
        y_tr = _load_y(data_root / "Y_train.txt").squeeze()

        X_te = _load_X(data_root / "X_test.txt")
        y_te = _load_y(data_root / "Y_test.txt").squeeze()

        self.train_ds = TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.long)
        )
        self.val_ds = TensorDataset(
            torch.tensor(X_te, dtype=torch.float32),
            torch.tensor(y_te, dtype=torch.long)
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


# ─────────────────────────────────────────────────────────────
# Model Components
# ─────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, Seq_Len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ActionClassificationTransformer(pl.LightningModule):
    def __init__(
        self,
        input_size: int = N_FEATURES,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        num_classes: int = 6,
        learning_rate: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        # 1. Patch Embedding
        self.input_proj = nn.Linear(input_size, d_model)

        # 2. Positional Encoding
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        # 3. Transformer Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            enc_layer, 
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )

        # 4. Classification Head
        self.classifier = nn.Linear(d_model, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer_encoder(x)
        # Global Average Pooling over temporal dimension (dim 1)
        x = x.mean(dim=1)
        return self.classifier(x)

    def _shared_step(self, batch, stage: str):
        X, y = batch
        logits = self(X)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc", acc, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }

    @torch.no_grad()
    def predict_label(self, pose_sequence: np.ndarray) -> str:
        self.eval()
        x = torch.tensor(pose_sequence, dtype=torch.float32).unsqueeze(0)
        # Ensure tensor is on the same device as the model
        x = x.to(self.device)
        logits = self(x)
        idx = logits.argmax(dim=1).item()
        return LABELS[idx]

# Backward-compatibility alias
ActionClassificationLSTM = ActionClassificationTransformer
