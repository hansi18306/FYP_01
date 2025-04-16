# Set TensorFlow log level and disable oneDNN custom operations before importing TensorFlow
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # 1 = Filter INFO, 2 = Filter INFO and WARNING


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
import warnings
from nltk.corpus import stopwords
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), './.env'))
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Download necessary resources from NLTK
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

# Set Tesseract executable path (update accordingly)
pytesseract.pytesseract.tesseract_cmd = r"model\Tesseract-OCR\tesseract.exe"  # Change path if needed

app = FastAPI(title="Cyberbullying Detection API", version="1.0")

# Load the trained model CNN
model = load_model('model/cyberbullying_CNN.keras')
# Load tokenizer object
import pickle
with open('model/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Initialize the Whisper pipeline for audio transcription
model_id = "openai/whisper-base"
whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    token=HUGGINGFACE_TOKEN
)
whisper_processor = AutoProcessor.from_pretrained(
    model_id,
    token=HUGGINGFACE_TOKEN
)
whisper_pipe = pipeline(
    "automatic-speech-recognition",
    model=whisper_model,
    tokenizer=whisper_processor.tokenizer,
    feature_extractor=whisper_processor.feature_extractor,
    device=device
)

class PredictionResponse(BaseModel):
    predicted_type: str

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

# new_text = input("Enter text to predict cyberbullying type: ")
# predicted_type = predict_cyberbullying_type(new_text)
# print("Predicted Cyberbullying Type:", predicted_type)

# Endpoints
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