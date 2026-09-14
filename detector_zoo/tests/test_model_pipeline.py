import os
import tempfile
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from src.model_pipeline import create_baseline_pipeline, BaselineDetector

def test_create_baseline_pipeline_returns_detector():
    # Act
    detector = create_baseline_pipeline(random_state=42)
    
    # Assert
    assert isinstance(detector, BaselineDetector)
    assert isinstance(detector.pipeline, Pipeline)
    assert len(detector.pipeline.steps) == 2
    assert detector.pipeline.steps[0][0] == 'tfidf'
    assert isinstance(detector.pipeline.steps[0][1], TfidfVectorizer)
    assert detector.pipeline.steps[1][0] == 'clf'
    assert isinstance(detector.pipeline.steps[1][1], LogisticRegression)

def test_pipeline_can_fit_and_predict():
    # Setup
    detector = create_baseline_pipeline(random_state=42)
    X_train = pd.Series(["free money now", "hello friend", "win a prize", "meeting at 10"])
    y_train = pd.Series([1, 0, 1, 0])
    
    # Act
    detector.fit(X_train, y_train)
    predictions = detector.predict(X_train)
    
    # Assert
    assert len(predictions) == 4
    assert set(predictions).issubset({0, 1})

def test_pipeline_logs_seed(capsys):
    # Act
    create_baseline_pipeline(random_state=123)
    
    # Assert
    captured = capsys.readouterr()
    assert "Initializing baseline pipeline with random_state=123" in captured.out

def test_model_persistence():
    """
    Tests saving and loading of the BaselineDetector, asserting that
    prediction probabilities are exactly identical upon loading.
    """
    # Create dummy data
    X_train = pd.Series(["buy now", "click here", "hello world", "how are you"])
    y_train = pd.Series([1, 1, 0, 0])
    
    # Train the dummy model
    model = BaselineDetector(random_state=42)
    model.fit(X_train, y_train)
    
    # Get original predictions
    X_test = pd.Series(["please click here to buy"])
    orig_probs = model.predict_proba(X_test)
    
    # Save model to a temporary directory
    with tempfile.TemporaryDirectory() as tempdir:
        filepath = os.path.join(tempdir, 'model.joblib')
        model.save_model(filepath)
        
        # Load into new instance
        loaded_model = BaselineDetector.load_model(filepath)
        
    # Get predictions from loaded model
    loaded_probs = loaded_model.predict_proba(X_test)
    
    # Assert exact match
    np.testing.assert_array_equal(orig_probs, loaded_probs)
