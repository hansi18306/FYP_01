# Set TensorFlow log level and disable oneDNN custom operations before importing TensorFlow
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # 1 = Filter INFO, 2 = Filter INFO and WARNING

import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Download necessary resources from NLTK
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

# Load the trained model and tokenizer
model = load_model('../model/cyberbullying_CNN.keras')
# Assuming you have saved the tokenizer object as 'tokenizer.pickle'
import pickle
with open('../model/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Slang dictionary for preprocessing
slang_dict = {
    'lmao': 'laughing my ass off',
    'k': 'okay',
    'y': 'why',
    'andme': 'and me',
    'ftsomething': 'face time something',
    'fkcn': 'fucking',
    '1 st': 'first',
    'init': 'is not it',
    'comp': 'compensation',
    'arr': 'arrive',
    'studs': 'students',
    'tho': 'though',
    'irl': 'in real life',
    'iykyk': 'if you know, you know',
    'fr': 'for real',
    'brb': 'be right back',
    'idk': 'i do not know',
    'imo': 'in my opinion',
    'omg': 'oh my god',
    'btw': 'by the way',
    'ttyl': 'talk to you later',
    'smh': 'shaking my head',
    'tbh': 'to be honest',
    'nvm': 'never mind',
    'gtg': 'got to go',
    'dm': 'direct message',
    'rn': 'right now',
    'np': 'no problem',
    'lol': 'laughing out loud',
    'pls': 'please',
    'omw': 'on my way',
    'fyi': 'for your information',
    'b4': 'before'
}

# Function to replace slang using slang_dict
def replace_slang(text):
    for word, replacement in slang_dict.items():
        text = re.sub(r'\b' + re.escape(word) + r'\b', replacement, text)
    return text

# Function for full text preprocessing
def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    # Replace slang
    text = replace_slang(text)
    # Remove numbers and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize text
    tokens = text.split()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize the tokens
    # tokens = [lemmatizer.lemmatize(word) for word in tokens]    
    # Join tokens back to a single string
    clean_text = ' '.join(tokens)
    # Function to remove words with less than 3 characters (except 'I' and 'a')
    clean_text = remove_short_words(clean_text)
    return clean_text

def remove_short_words(text):
    return ' '.join([word for word in text.split() if len(word) > 2 or word in ['i', 'a']])

# Define the predict_cyberbullying_type function
def predict_cyberbullying_type(text):
    # Preprocess the input text
    text_clean = preprocess_text(text)

    # Convert the text to a sequence of numerical tokens
    text_seq = tokenizer.texts_to_sequences([text_clean])

    # Pad the sequence to match the input length of the model
    text_pad = pad_sequences(text_seq, maxlen=300, padding='post', truncating='post')

    # Make a prediction using the model
    prediction = model.predict(text_pad)

    # Get the index of the predicted class
    predicted_class = np.argmax(prediction)

    # Map the predicted class index to the corresponding cyberbullying type
    cyberbullying_types = ['religion', 'age', 'gender', 'ethnicity', 'not_cyberbullying']
    predicted_type = cyberbullying_types[predicted_class]

    return predicted_type

new_text = input("Enter text to predict cyberbullying type: ")
predicted_type = predict_cyberbullying_type(new_text)
print("Predicted Cyberbullying Type:", predicted_type)