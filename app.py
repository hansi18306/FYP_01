from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from io import BytesIO
from pydub import AudioSegment
from PIL import Image
import pytesseract
import torchaudio
import torch
import pickle
import re
import warnings
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from transformers import pipeline

# Ensure required NLTK data is available
nltk.download('punkt')
nltk.download('stopwords')

# Set Tesseract executable path (update accordingly)
pytesseract.pytesseract.tesseract_cmd = r"model\Tesseract-OCR\tesseract.exe"  # Change path if needed

app = FastAPI(title="Cyberbullying Detection API", version="1.0")

# Load the saved logistic regression model and count vectorizer
try:
    with open('model/logistic_regression_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    with open('model/count_vectorizer.pkl', 'rb') as f:
        cv = pickle.load(f)
except Exception as e:
    raise Exception("Error loading model or vectorizer: " + str(e))

# Initialize the Whisper pipeline for audio transcription
device = "cuda:0" if torch.cuda.is_available() else "cpu"
whisper_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=device)

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
    return ' '.join(words)

def predict_cyberbullying_type(text: str) -> str:
    cleaned_text = clean_text(text)
    text_vector = cv.transform([cleaned_text])
    prediction = loaded_model.predict(text_vector)[0]
    return prediction

def transcribe_audio(waveform, sample_rate):
    # Convert waveform to mono (1D)
    if waveform.size(0) > 1:  # If stereo, average channels
        waveform = waveform.mean(dim=0)
    else:
        waveform = waveform.squeeze()
    audio_input = waveform.numpy()

    # Resample if needed (e.g., to 16000 Hz)
    desired_sample_rate = 16000
    if sample_rate != desired_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=desired_sample_rate)
        # Resampler expects at least 2D (channels, samples)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        waveform = resampler(waveform)
        # Ensure mono after resampling
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze()
        audio_input = waveform.numpy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        result = whisper_pipe(audio_input, return_timestamps=True)
    return result["text"]

@app.post("/predict_text", response_model=PredictionResponse)
def predict_text(text: str = Form(...)):
    if not text:
        raise HTTPException(status_code=400, detail="Input text is required")
    prediction = predict_cyberbullying_type(text)
    return PredictionResponse(predicted_type=prediction)

@app.post("/predict_audio", response_model=PredictionResponse)
async def predict_audio(file: UploadFile = File(...)):
    if file.content_type not in ["audio/mp3", "audio/mpeg", "audio/wav"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only mp3 or wav allowed.")
    try:
        contents = await file.read()
        file_extension = file.filename.split('.')[-1]
        audio = AudioSegment.from_file(BytesIO(contents), format=file_extension)
        sample_rate = audio.frame_rate

        # Normalize audio samples to [-1.0, 1.0].
        norm_factor = float(2 ** (8 * audio.sample_width - 1))
        if audio.channels > 1:
            channels = audio.split_to_mono()
            waveform = torch.stack([
                torch.tensor(ch.get_array_of_samples(), dtype=torch.float32)
                for ch in channels
            ])
        else:
            waveform = torch.tensor(audio.get_array_of_samples(), dtype=torch.float32).unsqueeze(0)
        waveform = waveform / norm_factor

        # Transcribe audio then predict based on transcription.
        transcription = transcribe_audio(waveform, sample_rate)
        prediction = predict_cyberbullying_type(transcription)
        return PredictionResponse(predicted_type=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/predict_image", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        extracted_text = pytesseract.image_to_string(image)
        prediction = predict_cyberbullying_type(extracted_text)
        return PredictionResponse(predicted_type=prediction)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)