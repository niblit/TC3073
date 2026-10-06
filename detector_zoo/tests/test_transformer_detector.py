import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.transformer_detector import TransformerDetector
from src.base_detector import BaseDetector


# ---- Fixtures ----

@pytest.fixture(scope="module")
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


@pytest.fixture(scope="module")
def trained_detector(small_dataset):
    """
    A TransformerDetector fine-tuned for just 1 epoch on the small dataset.
    Scoped to module so the expensive model download + training only happens once.
    """
    X, y = small_dataset
    detector = TransformerDetector(
        random_state=42,
        epochs=1,
        max_len=32,
        batch_size=4,
        warmup_steps=0,
    )
    detector.fit(X, y)
    return detector


# ---- Detector interface tests ----

def test_detector_is_base_detector():
    detector = TransformerDetector.__new__(TransformerDetector)
    # Even a bare instance should satisfy the type check
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
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_detector_logs_seed(capsys):
    TransformerDetector(random_state=77, epochs=1, max_len=16)
    captured = capsys.readouterr()
    assert "random_state=77" in captured.out
    assert "Transformer" in captured.out


# ---- Persistence tests ----

def test_model_persistence(small_dataset, trained_detector):
    """Saved and loaded detector must produce identical predictions."""
    X, _ = small_dataset
    orig_probs = trained_detector.predict_proba(X)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "transformer_model")
        trained_detector.save_model(save_path)

        assert os.path.isfile(os.path.join(save_path, "config.json"))
        assert os.path.isfile(os.path.join(save_path, "hparams.json"))

        loaded = TransformerDetector.load_model(save_path)

    loaded_probs = loaded.predict_proba(X)
    np.testing.assert_array_almost_equal(orig_probs, loaded_probs, decimal=5)
