import pandas as pd

def load_data(file_path):
    """Load and return the dataset from the specified file path."""
    return pd.read_csv(file_path)

def sample_data(data, sample_fraction=0.2, seed=42):
    """Sample the data."""
    return data.sample(frac=sample_fraction, random_state=seed)
