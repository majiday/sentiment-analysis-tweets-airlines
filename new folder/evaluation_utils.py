from sklearn.metrics import classification_report

def evaluate_performance(predictions, true_labels):
    """Calculate and print the classification report."""
    return classification_report(true_labels, predictions, labels=['negative', 'neutral', 'positive'])

def save_predictions(data, file_path):
    """Save the DataFrame to a CSV file."""
    data.to_csv(file_path, index=False)
