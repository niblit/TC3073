"""
Character-level CNN detector for phishing/BEC classification.

Architecture follows the spirit of Zhang et al. "Character-level Convolutional Networks
for Text Classification" (2015), adapted into a compact form suitable for binary
classification on email/phishing text.

The detector wraps the PyTorch model behind the same fit/predict/save/load interface
used by all detectors in the zoo, so it is a drop-in replacement for the baseline.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from src.base_detector import BaseDetector
from src.device import get_device

# ---------------------------------------------------------------------------
# Character vocabulary
# ---------------------------------------------------------------------------
# We encode each character as an integer index.  Characters outside this
# alphabet are mapped to 0 (the "unknown / padding" index).
ALPHABET = list("abcdefghijklmnopqrstuvwxyz0123456789 .,;:!?'\"-/\\|_@#$%^&*~`+=<>(){}[]")
CHAR_TO_IDX = {ch: i + 1 for i, ch in enumerate(ALPHABET)}  # 0 = pad/unknown
VOCAB_SIZE = len(ALPHABET) + 1  # +1 for pad/unknown index 0

DEFAULT_MAX_LEN = 1024      # max characters per sample
DEFAULT_EMBED_DIM = 32
DEFAULT_NUM_FILTERS = 64
DEFAULT_KERNEL_SIZES = [3, 5, 7]
DEFAULT_DROPOUT = 0.3
DEFAULT_LR = 1e-3
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 64


def _encode_text(text: str, max_len: int = DEFAULT_MAX_LEN) -> np.ndarray:
    """Encode a single text string into a fixed-length integer array."""
    text = text.lower()
    encoded = np.zeros(max_len, dtype=np.int64)
    for i, ch in enumerate(text[:max_len]):
        encoded[i] = CHAR_TO_IDX.get(ch, 0)
    return encoded


def _encode_batch(texts, max_len: int = DEFAULT_MAX_LEN) -> np.ndarray:
    """Encode a batch of text strings."""
    return np.stack([_encode_text(t, max_len) for t in texts])


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class _TextDataset(Dataset):
    def __init__(self, X_encoded: np.ndarray, y: np.ndarray | None = None):
        self.X = torch.from_numpy(X_encoded).long()
        self.y = torch.from_numpy(y).float() if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


# ---------------------------------------------------------------------------
# PyTorch model
# ---------------------------------------------------------------------------

class CharCNNModel(nn.Module):
    """
    Multi-kernel character-level CNN.

    For each kernel size we run a 1-d convolution over the character-embedding
    sequence, apply ReLU + global max-pooling, concatenate the features from
    all kernels, and pass through a small fully-connected head.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = DEFAULT_EMBED_DIM,
        num_filters: int = DEFAULT_NUM_FILTERS,
        kernel_sizes: list | None = None,
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__()
        kernel_sizes = kernel_sizes or DEFAULT_KERNEL_SIZES

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k, padding=k // 2)
            for k in kernel_sizes
        ])

        total_filters = num_filters * len(kernel_sizes)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(total_filters, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, max_len)
        emb = self.embedding(x)          # (batch, max_len, embed_dim)
        emb = emb.transpose(1, 2)        # (batch, embed_dim, max_len)

        conv_outs = []
        for conv in self.convs:
            c = torch.relu(conv(emb))    # (batch, num_filters, seq_len')
            c = c.max(dim=2).values      # (batch, num_filters)  – global max-pool
            conv_outs.append(c)

        cat = torch.cat(conv_outs, dim=1)  # (batch, total_filters)
        logits = self.fc(cat).squeeze(-1)  # (batch,)
        return logits


# ---------------------------------------------------------------------------
# Detector wrapper
# ---------------------------------------------------------------------------

class CharCNNDetector(BaseDetector):
    """
    Character-level CNN detector.

    Wraps a PyTorch CharCNNModel behind the standard BaseDetector interface
    so it can be used interchangeably with the baseline logistic-regression
    detector.
    """

    def __init__(
        self,
        random_state: int = 42,
        max_len: int = DEFAULT_MAX_LEN,
        embed_dim: int = DEFAULT_EMBED_DIM,
        num_filters: int = DEFAULT_NUM_FILTERS,
        kernel_sizes: list | None = None,
        dropout: float = DEFAULT_DROPOUT,
        lr: float = DEFAULT_LR,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        super().__init__(random_state)
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes or DEFAULT_KERNEL_SIZES
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.device = get_device()

        self.model = CharCNNModel(
            vocab_size=VOCAB_SIZE,
            embed_dim=self.embed_dim,
            num_filters=self.num_filters,
            kernel_sizes=self.kernel_sizes,
            dropout=self.dropout,
        ).to(self.device)

        print(
            f"Initializing CharCNN pipeline with random_state={self.random_state}, "
            f"max_len={self.max_len}, epochs={self.epochs}, device={self.device}"
        )

    # ------------------------------------------------------------------
    # fit / predict / predict_proba
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """Train the character-level CNN on raw text data *X* with labels *y*."""
        X_enc = _encode_batch(X, self.max_len)
        y_arr = np.asarray(y, dtype=np.float32)

        dataset = _TextDataset(X_enc, y_arr)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                logits = self.model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch_X.size(0)

            avg_loss = total_loss / len(dataset)
            print(f"  [CharCNN] Epoch {epoch + 1}/{self.epochs}  loss={avg_loss:.4f}")

        return self

    def predict(self, X) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X) -> np.ndarray:
        X_enc = _encode_batch(X, self.max_len)
        dataset = _TextDataset(X_enc)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        all_probs = []
        with torch.no_grad():
            for batch_X in loader:
                if isinstance(batch_X, (list, tuple)):
                    batch_X = batch_X[0]
                batch_X = batch_X.to(self.device)
                logits = self.model(batch_X)
                prob_pos = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(prob_pos)

        prob_pos = np.concatenate(all_probs)
        prob_neg = 1.0 - prob_pos
        return np.column_stack([prob_neg, prob_pos])

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def save_model(self, filepath: str):
        """Save the trained CharCNN model and its hyperparameters to *filepath*."""
        os.makedirs(filepath, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(filepath, "model.pt"))
        hparams = {
            "random_state": self.random_state,
            "max_len": self.max_len,
            "embed_dim": self.embed_dim,
            "num_filters": self.num_filters,
            "kernel_sizes": self.kernel_sizes,
            "dropout": self.dropout,
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
        }
        with open(os.path.join(filepath, "hparams.json"), "w") as f:
            json.dump(hparams, f, indent=2)

    @classmethod
    def load_model(cls, filepath: str) -> 'CharCNNDetector':
        """Load a trained CharCNN model from *filepath*."""
        with open(os.path.join(filepath, "hparams.json")) as f:
            hparams = json.load(f)

        instance = cls(**hparams)
        state = torch.load(
            os.path.join(filepath, "model.pt"),
            map_location=instance.device,
            weights_only=True,
        )
        instance.model.load_state_dict(state)
        return instance
