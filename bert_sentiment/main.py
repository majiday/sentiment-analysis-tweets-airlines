from data_processing import load_data, sample_data, extract_column
from sentiment_analysis import initialize_model, get_predictions, map_sentiment, calculate_accuracy

def main():
    # Load and sample data
    data = load_data('./Tweets.csv')
    sampled_data = sample_data(data, fraction=0.2, seed=42)

    # Extract tweets and human labels
    tweets = extract_column(sampled_data, 'text')
    human_labels = extract_column(sampled_data, 'airline_sentiment')
    human_labels = [label.lower() for label in human_labels]

    # Initialize model and get predictions
    model = initialize_model("nlptown/bert-base-multilingual-uncased-sentiment")
    predictions = get_predictions(model, tweets)

    # Process model predictions
    model_sentiments = [map_sentiment(pred['label']) for pred in predictions]

    # Calculate accuracy
    accuracy = calculate_accuracy(model_sentiments, human_labels)
    print(f"Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()
