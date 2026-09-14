import json
import pandas as pd
import os

def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads JSON data containing 'text' and 'label' fields into a Pandas DataFrame.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
        
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    df = pd.DataFrame(data)
    
    # Ensure correct columns if the dataframe is not empty
    if not df.empty:
        assert 'text' in df.columns and 'label' in df.columns, "JSON must contain 'text' and 'label' keys."
    else:
        # If it is empty, explicitly set the expected columns
        df = pd.DataFrame(columns=['text', 'label'])
        
    return df
