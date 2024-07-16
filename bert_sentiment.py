# Install required packages
    #!pip install pandas transformers torch

# Import required packages
import pandas as pd
from transformers import pipeline


# Function to map the star rating to positive, negative, and neutral
def map_sentiment(label):
    stars = int(label.split()[0])  
    if stars in [1, 2]:
        return 'negative'
    elif stars == 3:
        return 'neutral'
    elif stars in [4, 5]:
        return 'positive'

# Load the dataset
file_path = './Tweets.csv'  
data = pd.read_csv(file_path)

# Sample 20% of the data
sampled_data = data.sample(frac=0.2, random_state=42)

# Extract tweets
tweets = sampled_data['text'].tolist()

# Initialize the sentiment analysis pipeline
model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Analyze sentiments of sampled tweets
predictions = model(tweets)

# Extract model sentiments and map them to three categories
model_sentiments = [map_sentiment(pred['label']) for pred in predictions]

# Assuming human_labels are loaded and correspond to the tweets' order
human_labels = sampled_data['airline_sentiment'].str.lower().tolist()

# Calculate accuracy
accuracy = sum(1 for model, human in zip(model_sentiments, human_labels) if model == human) / len(human_labels)

print(f"Accuracy: {accuracy:.2f}")
