from transformers import pipeline
import torch

def initialize_model(device=-1):
    """Initialize the sentiment analysis model with optional GPU support."""
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment", device=device)

def predict_sentiments(model, tweets):
    """Use the model to predict sentiments of tweets."""
    predictions = model(tweets)
    label_mapping = {
        'LABEL_0': 'negative',
        'LABEL_1': 'neutral',
        'LABEL_2': 'positive'
    }
    return [label_mapping[pred['label']] for pred in predictions]
