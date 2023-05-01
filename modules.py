import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import joblib

# Define a function to preprocess the text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if not token in stop_words]
    
    # Apply stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    
    # Join tokens into a string
    text = ' '.join(tokens)
    
    return text

def model_create(data):
    data=pd.read_csv(data)
    # Apply the preprocessing function to the content column
    data['content'] = data['content'].apply(preprocess_text)
    train_data, test_data, train_labels, test_labels = train_test_split(data['content'], data['reply_to_id'], test_size=0.2, random_state=42)    
    vectorizer = CountVectorizer()
    train_vectors = vectorizer.fit_transform(train_data)
    test_vectors = vectorizer.transform(test_data)
    clf = MultinomialNB()
    clf.fit(train_vectors, train_labels)
    joblib.dump(clf, 'classifier.joblib')
    joblib.dump(vectorizer, 'vectorizer.joblib')
    return clf.score(test_vectors, test_labels)

def predict(data):
    data=preprocess_text(data)
    vectorizer = joblib.load('vectorizer.joblib')
    clf = joblib.load('classifier.joblib')
    new_vector = vectorizer.transform([data])
    predicted_label = clf.predict(new_vector)[0]
    return predicted_label


