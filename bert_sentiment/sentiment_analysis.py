from transformers import pipeline

def initialize_model(model_name):
    """ Initialize sentiment analysis pipeline. """
    return pipeline("sentiment-analysis", model=model_name)

def get_predictions(model, data):
    """ Get model predictions for the provided data. """
    return model(data)

def map_sentiment(label):
    """ Map numeric star rating to sentiment categories. """
    stars = int(label.split()[0])
    if stars in [1, 2]:
        return 'negative'
    elif stars == 3:
        return 'neutral'
    elif stars in [4, 5]:
        return 'positive'

def calculate_accuracy(model_sentiments, human_labels):
    """ Calculate the accuracy of model predictions. """
    correct = sum(1 for model, human in zip(model_sentiments, human_labels) if model == human)
    return correct / len(human_labels)
