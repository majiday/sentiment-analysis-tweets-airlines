import openai

def initialize_client(api_key_path='api_key.txt'):
    with open(api_key_path) as api_key:
        openai_key = api_key.readline().strip()
    return openai.OpenAI(api_key=openai_key)

def analyze_sentiment(df, client):
    results = []
    total = 0
    matches = 0
    for index, row in df.iterrows():
        tweet = row['text']
        original_sentiment = row['airline_sentiment']
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI trained to categorize the sentiment of tweets strictly as 'positive', 'negative', or 'neutral'."},
                {"role": "user", "content": f"Analyze and categorize the sentiment of this tweet strictly as 'positive', 'negative', or 'neutral': '{tweet}'"}
            ]
        )
        ai_sentiment = completion.choices[0].message.content.strip().lower()
        sentiment_word = 'positive' if 'positive' in ai_sentiment else 'negative' if 'negative' in ai_sentiment else 'neutral' if 'neutral' in ai_sentiment else 'unclear'
        results.append((row['tweet_id'], original_sentiment, sentiment_word))
        total += 1
        if sentiment_word == original_sentiment:
            matches += 1
    match_rate = (matches / total) * 100 if total > 0 else 0
    return results, match_rate
