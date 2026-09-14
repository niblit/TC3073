import pandas as pd
import pytest
from src.splitter import split_data

def test_split_data_returns_correct_splits():
    # Setup
    df = pd.DataFrame({
        'text': [f'email {i}' for i in range(10)],
        'label': [i % 2 for i in range(10)]
    })
    
    # Act
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=42)
    
    # Assert
    assert len(X_train) == 8
    assert len(X_test) == 2
    assert len(y_train) == 8
    assert len(y_test) == 2
    
def test_split_data_reproducibility():
    # Setup
    df = pd.DataFrame({
        'text': [f'msg {i}' for i in range(100)],
        'label': [i % 2 for i in range(100)]
    })
    
    # Act
    X_train1, X_test1, _, _ = split_data(df, test_size=0.25, random_state=123)
    X_train2, X_test2, _, _ = split_data(df, test_size=0.25, random_state=123)
    
    # Assert: Exact same indices should be selected
    assert X_train1.equals(X_train2)
    assert X_test1.equals(X_test2)

def test_split_data_logs_seed(capsys):
    # Setup
    df = pd.DataFrame({
        'text': [f'msg {i}' for i in range(10)],
        'label': [i % 2 for i in range(10)]
    })
    
    # Act
    split_data(df, test_size=0.2, random_state=42)
    
    # Assert
    captured = capsys.readouterr()
    assert "Using random_state=42 for data splitting" in captured.out
