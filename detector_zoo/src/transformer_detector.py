"""
Fine-tuned small transformer detector for phishing/BEC classification.

Uses DistilBERT (distilbert-base-uncased) — a compact, 6-layer distillation of
BERT that retains ~97 % of BERT's language-understanding capability at roughly
half the size and twice the speed.

The detector wraps the Hugging Face Trainer behind the standard BaseDetector
interface so it is a drop-in replacement for the baseline.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from src.base_detector import BaseDetector
from src.device import get_device

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL_NAME = "distilbert-base-uncased"
DEFAULT_MAX_LEN = 256
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 16
DEFAULT_LR = 2e-5
DEFAULT_WARMUP_STEPS = 50


# ---------------------------------------------------------------------------
# PyTorch Dataset for HF Trainer
# ---------------------------------------------------------------------------

class _HFDataset(Dataset):
    """Thin wrapper that holds tokenised inputs + labels for the HF Trainer."""

    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ---------------------------------------------------------------------------
# Detector wrapper
# ---------------------------------------------------------------------------

class TransformerDetector(BaseDetector):
    """
    Fine-tuned DistilBERT detector.

    On *fit*, the pre-trained model is fine-tuned via the Hugging Face Trainer.
    On *predict* / *predict_proba*, the model runs inference and returns
    labels / probabilities consistent with the BaseDetector API.
    """

    def __init__(
        self,
        random_state: int = 42,
        model_name: str = DEFAULT_MODEL_NAME,
        max_len: int = DEFAULT_MAX_LEN,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        lr: float = DEFAULT_LR,
        warmup_steps: int = DEFAULT_WARMUP_STEPS,
    ):
        super().__init__(random_state)
        self.model_name = model_name
        self.max_len = max_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.warmup_steps = warmup_steps

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.device = get_device()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2,
        )

        print(
            f"Initializing Transformer pipeline ({self.model_name}) with "
            f"random_state={self.random_state}, max_len={self.max_len}, "
            f"epochs={self.epochs}, device={self.device}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tokenize(self, texts):
        """Tokenize a list/Series of strings using the HF tokenizer."""
        return self.tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

    # ------------------------------------------------------------------
    # fit / predict / predict_proba
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """Fine-tune the transformer on raw text *X* with labels *y*."""
        encodings = self._tokenize(X)
        labels = np.asarray(y, dtype=np.int64)
        dataset = _HFDataset(encodings, labels)

        training_args = TrainingArguments(
            output_dir=os.path.join("models", "_transformer_training_tmp"),
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            learning_rate=self.lr,
            warmup_steps=self.warmup_steps,
            weight_decay=0.01,
            logging_steps=10,
            save_strategy="no",          # we save manually via save_model()
            seed=self.random_state,
            report_to="none",            # disable wandb / mlflow
            disable_tqdm=False,
            dataloader_pin_memory=(self.device.type == "cuda"),
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()
        self.model.to(self.device)
        return self

    def predict(self, X) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X) -> np.ndarray:
        encodings = self._tokenize(X)
        dataset = _HFDataset(encodings)

        self.model.eval()
        self.model.to(self.device)

        all_probs = []
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)

        return np.concatenate(all_probs)

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def save_model(self, filepath: str):
        """Save the fine-tuned model, tokenizer, and hyperparameters."""
        os.makedirs(filepath, exist_ok=True)
        self.model.save_pretrained(filepath)
        self.tokenizer.save_pretrained(filepath)
        hparams = {
            "random_state": self.random_state,
            "model_name": self.model_name,
            "max_len": self.max_len,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "warmup_steps": self.warmup_steps,
        }
        with open(os.path.join(filepath, "hparams.json"), "w") as f:
            json.dump(hparams, f, indent=2)

    @classmethod
    def load_model(cls, filepath: str) -> 'TransformerDetector':
        """Load a fine-tuned transformer model from *filepath*."""
        with open(os.path.join(filepath, "hparams.json")) as f:
            hparams = json.load(f)

        # Create the instance *without* downloading the base weights again —
        # we'll overwrite the model with the saved fine-tuned weights.
        instance = cls.__new__(cls)
        instance.random_state = hparams["random_state"]
        instance.model_name = hparams["model_name"]
        instance.max_len = hparams["max_len"]
        instance.epochs = hparams["epochs"]
        instance.batch_size = hparams["batch_size"]
        instance.lr = hparams["lr"]
        instance.warmup_steps = hparams["warmup_steps"]
        instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        instance.tokenizer = AutoTokenizer.from_pretrained(filepath)
        instance.model = AutoModelForSequenceClassification.from_pretrained(filepath)
        instance.model.to(instance.device)
        return instance
