import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

def split_data(
    df: pd.DataFrame, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Splits the dataframe into training and testing sets for text and label.
    Explicitly logs the random seed used for reproducibility.
    """
    print(f"Using random_state={random_state} for data splitting (test_size={test_size})")
    
    X = df['text']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # Ensure balanced class distribution in splits
    )
    
    return X_train, X_test, y_train, y_test
