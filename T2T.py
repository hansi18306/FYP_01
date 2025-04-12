from fastapi import FastAPI, HTTPException, Form
from pydantic import BaseModel
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

# Ensure required NLTK data is available
nltk.download('punkt')
nltk.download('stopwords')

# Load the saved model and count vectorizer
try:
    with open('model/logistic_regression_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    with open('model/count_vectorizer.pkl', 'rb') as f:
        cv = pickle.load(f)
except Exception as e:
    raise Exception("Error loading model or vectorizer: " + str(e))

app = FastAPI(title="Cyberbullying Detection API", version="1.0")

class PredictionResponse(BaseModel):
    predicted_type: str

def clean_text(text: str) -> str:
    # Remove HTML tags
    text = re.sub('<.*?>', '', text)
    # Remove non-alphabetic characters and convert to lowercase
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    # Remove URLs, mentions, and hashtags
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'#\S+', '', text)
    # Tokenize the text
    words = nltk.word_tokenize(text)
    # Remove stopwords
    words = [w for w in words if w not in stopwords.words('english')]
    # Stem the words
    stemmer = PorterStemmer()
    words = [stemmer.stem(w) for w in words]
    # Join the words back into a string
    return ' '.join(words)

def predict_cyberbullying_type(text: str) -> str:
    cleaned_text = clean_text(text)
    text_vector = cv.transform([cleaned_text])
    prediction = loaded_model.predict(text_vector)[0]
    return prediction

@app.post("/predict_bulling", response_model=PredictionResponse)
def predict_bulling(text: str = Form(...)):
    if not text:
        raise HTTPException(status_code=400, detail="Input text is required")
    predicted = predict_cyberbullying_type(text)
    return PredictionResponse(predicted_type=predicted)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)