from data_processing import load_and_sample_data
from sentiment_analysis import initialize_client, analyze_sentiment
from results_reporting import save_results, generate_report

def main():
    df = load_and_sample_data('./Tweets.csv')
    client = initialize_client()
    results, match_rate = analyze_sentiment(df, client)
    print(f"Sentiment analysis completed. Match rate: {match_rate:.2f}%")
    results_df = save_results(results)
    generate_report(results_df)

if __name__ == "__main__":
    main()
