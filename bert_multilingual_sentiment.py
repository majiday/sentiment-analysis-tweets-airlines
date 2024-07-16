# install required packages
    #pip install transformers pandas

# Import required packages
import pandas as pd
from transformers import pipeline

# Load the dataset
file_path = './Tweets.csv'  
data = pd.read_csv(file_path)

# Sample 20% of the data
sampled_data = data.sample(frac=0.2, random_state=42)

# Extract tweets
tweets = sampled_data['text'].tolist()

# Initialize the sentiment analysis pipeline from Hugging Face
model = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

# Analyze sentiments of sampled tweets
predictions = model(tweets)


# Convert model labels to text labels
label_mapping = {
    'LABEL_0': 'negative',
    'LABEL_1': 'neutral',
    'LABEL_2': 'positive'
}

# Extract model sentiments
model_sentiments = [label_mapping[pred['label']] for pred in predictions]

# Compare model predictions with human labels
human_labels = sampled_data['airline_sentiment'].str.lower().tolist()

# Calculate the accuracy
accuracy = sum(1 for model, human in zip(model_sentiments, human_labels) if model == human) / len(human_labels)

print(f"Accuracy: {accuracy:.2f}")


