import json
import pandas as pd
import pytest
from src.data_loader import load_data

def test_load_data_returns_dataframe(tmp_path):
    # Setup: Create a temporary dummy dataset
    dummy_data = [
        {"text": "safe email", "label": 0},
        {"text": "phishing email", "label": 1}
    ]
    file_path = tmp_path / "dummy_data.json"
    file_path.write_text(json.dumps(dummy_data))
    
    # Act
    df = load_data(str(file_path))
    
    # Assert
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == ["text", "label"]
    assert df.iloc[0]["text"] == "safe email"
    assert df.iloc[0]["label"] == 0

def test_load_data_handles_empty_file(tmp_path):
    # Setup
    file_path = tmp_path / "empty_data.json"
    file_path.write_text("[]")
    
    # Act
    df = load_data(str(file_path))
    
    # Assert
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert list(df.columns) == ["text", "label"]

def test_load_data_missing_file():
    with pytest.raises(FileNotFoundError):
        load_data("non_existent_file.json")
