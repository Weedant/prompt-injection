
# ============================================================================
# src/data_pipeline/loader.py
# ============================================================================
import pandas as pd
from typing import Tuple
import os

DATA_PATH = "data/processed/filtered_data.csv"

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load filtered_data.csv and split into train/test (80/20 stratified).
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if not {'id', 'text', 'label'}.issubset(df.columns):
        raise ValueError("Missing required columns: id, text, label")

    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df['label'], random_state=42
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
