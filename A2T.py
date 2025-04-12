from io import BytesIO
from pydub import AudioSegment
import warnings

from fastapi import FastAPI, UploadFile, File, HTTPException
import torchaudio
import torch
from transformers import pipeline

app = FastAPI()

# Set device and load the Whisper model once at startup
device = "cuda:0" if torch.cuda.is_available() else "cpu"
pipe = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=device)

def transcribe_audio(waveform, sample_rate):
    # Ensure the waveform is mono (1D)
    if waveform.size(0) > 1:  # If stereo, average the channels
        audio_input = waveform.mean(dim=0).numpy()
    else:
        audio_input = waveform.numpy()

    # Resample if needed (e.g., to 16000 Hz)
    desired_sample_rate = 16000
    if sample_rate != desired_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=desired_sample_rate)
        # Resample each channel separately then average if needed
        waveform = resampler(waveform)
        audio_input = waveform.mean(dim=0).numpy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        # Process the audio directly without `return_tensors`
        result = pipe(audio_input, return_timestamps=True)

    return result["text"]

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if file.content_type not in ["audio/mp3", "audio/mpeg", "audio/wav"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only mp3 or wav allowed.")
    try:
        contents = await file.read()
        # Extract file extension for pydub
        file_extension = file.filename.split('.')[-1]
        # Load audio in memory using pydub
        audio = AudioSegment.from_file(BytesIO(contents), format=file_extension)
        sample_rate = audio.frame_rate

        # Convert AudioSegment samples to torch.Tensor
        # pydub returns samples as integers; normalize to [-1.0, 1.0]
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

        transcription = transcribe_audio(waveform, sample_rate)
        return {"transcription": transcription}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)