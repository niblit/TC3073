import pytest
from src.metrics import evaluate_model

def test_evaluate_model_returns_correct_metrics():
    # Setup
    y_true = [0, 1, 0, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 0, 1] # 1 false negative
    
    # Act
    metrics = evaluate_model(y_true, y_pred)
    
    # Assert
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    
    assert metrics['accuracy'] == 5/6
    assert metrics['precision'] == 1.0  # (2 TP) / (2 TP + 0 FP)
    assert metrics['recall'] == 2/3     # (2 TP) / (2 TP + 1 FN)
    assert metrics['f1'] == 0.8         # 2 * (1.0 * 2/3) / (1.0 + 2/3) = 0.8
