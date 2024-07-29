import pandas as pd

def load_data(file_path):
    """ Load data from a CSV file. """
    return pd.read_csv(file_path)

def sample_data(data, fraction, seed):
    """ Sample a fraction of data for processing. """
    return data.sample(frac=fraction, random_state=seed)

def extract_column(data, column_name):
    """ Extract a column from the DataFrame as a list. """
    return data[column_name].tolist()
