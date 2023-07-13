#importing all the libraries
import pandas as pd
import re
import nltk
import joblib
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# loading the dataset usning pandas
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# some text preprocessing before applying the model
def preprocess_text(text):
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Lowercase
    text = text.lower()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Remove digits
    tokens = [token for token in tokens if not token.isdigit()]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Remove empty tokens and single meaningless letters
    tokens = [token for token in tokens if len(token) > 1]
    
    # Remove spaces
    tokens = [token.strip() for token in tokens]
    
    return tokens

def train_word2vec(data):
    # Preprocess text data
    text1_list = data['text1'].apply(preprocess_text).tolist()
    text2_list = data['text2'].apply(preprocess_text).tolist()

   # combinning all the words in 2d lists
    sentences = text1_list + text2_list

    # Train Word2Vec model
    model = Word2Vec(sentences, min_count=1)

    return model

def calculate_similarity(text1, text2, model):
    # Preprocess text1 and text2
    text1 = preprocess_text(text1)
    text2 = preprocess_text(text2)

    # Get Word2Vec embeddings for text1 and text2
    vec1 = [model.wv[word] for word in text1 if word in model.wv]
    vec2 = [model.wv[word] for word in text2 if word in model.wv]

    # Calculate average for the vector representation
    avg_vec1 = sum(vec1) / len(vec1) if vec1 else []
    avg_vec2 = sum(vec2) / len(vec2) if vec2 else []

    # Calculate cosine similarity
    similarity = cosine_similarity([avg_vec1], [avg_vec2])[0][0]

    return similarity

# Load and preprocess the data
data = load_data('Precily_Text_Similarity.csv')

# Train the Word2Vec model
word2vec_model = train_word2vec(data)


# Save the trained Word2Vec model using joblib
joblib.dump(word2vec_model, 'word2vec_model.pkl')


