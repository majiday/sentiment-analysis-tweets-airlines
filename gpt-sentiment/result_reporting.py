import pandas as pd
from sklearn.metrics import classification_report

def save_results(results, filename='results.csv'):
    results_df = pd.DataFrame(results, columns=['tweet_id', 'original_sentiment', 'sentiment_word'])
    results_df.to_csv(filename, index=False)
    return results_df

def generate_report(results_df):
    report = classification_report(results_df['original_sentiment'], results_df['sentiment_word'], labels=["positive", "negative", "neutral"], output_dict=True)
    df_metrics = pd.DataFrame(report).transpose()
    print(df_metrics)
