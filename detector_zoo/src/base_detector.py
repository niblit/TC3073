import joblib
from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class BaseDetector(ABC):
    """
    Abstract base class for all detectors in the zoo.
    Every detector must implement fit, predict, predict_proba, save_model, and load_model.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    @abstractmethod
    def fit(self, X, y) -> 'BaseDetector':
        """Train the detector on the given data."""
        ...

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """Return binary predictions."""
        ...

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray:
        """Return probability estimates for each class."""
        ...

    @abstractmethod
    def save_model(self, filepath: str) -> None:
        """Persist the trained model to disk."""
        ...

    @classmethod
    @abstractmethod
    def load_model(cls, filepath: str) -> 'BaseDetector':
        """Load a trained model from disk."""
        ...
