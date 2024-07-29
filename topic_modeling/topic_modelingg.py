from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def lda_for_entire_dataset(clean_text):
    return lda_topic_modeling(clean_text)

def lda_for_airline(tweets_df, airline):
    airline_df = tweets_df[tweets_df['airline'] == airline].copy()
    return lda_topic_modeling(airline_df['clean_text'])

def lda_topic_modeling(text_data):
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    dtm = vectorizer.fit_transform(text_data)
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(dtm)
    return display_topics(lda, vectorizer.get_feature_names_out(), 10)

def display_topics(model, feature_names, no_top_words):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        topics.append([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])
    return topics
