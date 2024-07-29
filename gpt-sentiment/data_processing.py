import pandas as pd

def load_and_sample_data(filepath, sample_size=200, seed=42):
    df_total = pd.read_csv(filepath)
    df_sample = df_total.sample(n=sample_size, random_state=seed)
    return df_sample[['text', 'tweet_id', 'airline_sentiment']]
