import pandas as pd
from text_processing import preprocess_dataset
from topic_modeling import lda_for_entire_dataset, lda_for_airline
from visualization import create_word_clouds

# Load the dataset
file_path = './Tweets.csv'
tweets_df = pd.read_csv(file_path)

# Preprocess the entire dataset
tweets_df['clean_text'] = preprocess_dataset(tweets_df['text'])

# Perform topic modeling for the entire dataset
print("Word Clouds for the Entire Dataset")
entire_dataset_topics = lda_for_entire_dataset(tweets_df['clean_text'])
create_word_clouds("Entire_Dataset", entire_dataset_topics)

# Perform topic modeling for each individual airline
subset_airlines = ['Virgin America', 'United', 'Southwest', 'Delta', 'US Airways', 'American']
airline_topics = {}
for airline in subset_airlines:
    print(f"Word Clouds for {airline}")
    airline_topics[airline] = lda_for_airline(tweets_df, airline)
    create_word_clouds(airline, airline_topics[airline])
