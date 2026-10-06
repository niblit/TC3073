import os
import tempfile
import numpy as np
import pandas as pd
import pytest
import torch

from src.char_cnn import (
    CharCNNDetector,
    CharCNNModel,
    _encode_text,
    _encode_batch,
    VOCAB_SIZE,
    DEFAULT_MAX_LEN,
)
from src.base_detector import BaseDetector


# ---- Fixtures ----

@pytest.fixture
def small_dataset():
    """Tiny dataset for fast unit tests."""
    X = pd.Series([
        "free money now click here",
        "hello friend how are you",
        "win a million dollar prize",
        "meeting scheduled at 10am",
        "urgent: verify your account",
        "lunch tomorrow at noon?",
    ])
    y = pd.Series([1, 0, 1, 0, 1, 0])
    return X, y


@pytest.fixture
def trained_detector(small_dataset):
    """A CharCNNDetector trained for just 2 epochs on the small dataset."""
    X, y = small_dataset
    detector = CharCNNDetector(random_state=42, epochs=2, max_len=64, batch_size=4)
    detector.fit(X, y)
    return detector


# ---- Encoding tests ----

def test_encode_text_length():
    encoded = _encode_text("hello", max_len=32)
    assert encoded.shape == (32,)
    assert encoded[0] != 0   # 'h' should map to a nonzero index
    assert encoded[5] == 0   # beyond the text → padding


def test_encode_text_unknown_chars():
    """Characters outside the alphabet should map to 0."""
    encoded = _encode_text("こんにちは", max_len=16)
    assert np.all(encoded == 0)


def test_encode_batch_shape():
    texts = ["aaa", "bbb", "ccc"]
    batch = _encode_batch(texts, max_len=16)
    assert batch.shape == (3, 16)


# ---- Model architecture tests ----

def test_charcnn_model_forward_shape():
    model = CharCNNModel(vocab_size=VOCAB_SIZE, embed_dim=16, num_filters=8, kernel_sizes=[3, 5])
    x = torch.randint(0, VOCAB_SIZE, (4, 64))
    logits = model(x)
    assert logits.shape == (4,)


# ---- Detector interface tests ----

def test_detector_is_base_detector():
    detector = CharCNNDetector(random_state=42, epochs=1)
    assert isinstance(detector, BaseDetector)


def test_detector_fit_and_predict(small_dataset, trained_detector):
    X, y = small_dataset
    predictions = trained_detector.predict(X)
    assert len(predictions) == len(X)
    assert set(predictions).issubset({0, 1})


def test_detector_predict_proba_shape(small_dataset, trained_detector):
    X, _ = small_dataset
    probs = trained_detector.predict_proba(X)
    assert probs.shape == (len(X), 2)
    # Probabilities should sum to 1 per row
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_detector_logs_seed(capsys):
    CharCNNDetector(random_state=99, epochs=1)
    captured = capsys.readouterr()
    assert "random_state=99" in captured.out
    assert "CharCNN" in captured.out


# ---- Persistence tests ----

def test_model_persistence(small_dataset, trained_detector):
    """Saved and loaded detector must produce identical predictions."""
    X, _ = small_dataset
    orig_probs = trained_detector.predict_proba(X)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "charcnn_model")
        trained_detector.save_model(save_path)

        assert os.path.isfile(os.path.join(save_path, "model.pt"))
        assert os.path.isfile(os.path.join(save_path, "hparams.json"))

        loaded = CharCNNDetector.load_model(save_path)

    loaded_probs = loaded.predict_proba(X)
    np.testing.assert_array_almost_equal(orig_probs, loaded_probs, decimal=5)
