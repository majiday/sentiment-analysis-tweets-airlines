import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english')) | set(['the', 'a', 'and', 'is', 'in', 'it', 'of', 'to', 'for', 'on', 'with', 'that', 'this', 'at', 'as', 'was', 'but', 'by', 'an'])
punctuation = set(string.punctuation)
lemmatizer = WordNetLemmatizer()

def preprocess_dataset(text_data):
    return text_data.apply(enhanced_preprocess_text)

def enhanced_preprocess_text(text):
    text = re.sub(r'@\w+', '', text)
    text = text.lower()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and not any(char in punctuation for char in word)]
    return ' '.join(tokens)
