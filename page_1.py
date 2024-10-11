import streamlit as st
from io import StringIO
import os 
from pydub import AudioSegment
from pathlib import Path
import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import pipeline

dirname = os.path.dirname(__file__)
model_path = os.path.join(dirname,"whisper_model_hi")

st.title("Audio Analytics")
uploaded_file = st.file_uploader(label="Choose an audio file",type=['wav',"mp3"])

if uploaded_file is not None:
    save_dir = "audio_files"
    file_details = {"filename":uploaded_file.name, "filetype":uploaded_file.type,"filesize":uploaded_file.size}
    if uploaded_file.name.endswith('wav'):
        audio = AudioSegment.from_wav(uploaded_file)
        file_type = 'wav'
    elif uploaded_file.name.endswith('mp3'):
        audio = AudioSegment.from_mp3(uploaded_file)
        file_type = 'mp3'
    save_path = os.path.join(Path(save_dir), uploaded_file.name)
    audio.export(save_path, format=file_type)
    
    # Audio Playing
    st.subheader("Audio Player")
    st.audio(save_path, format=file_type, loop=True)
    
    # Transcription 
    processor = WhisperProcessor.from_pretrained("openai/whisper-small.en")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small.en")

    audio_array, sampling_rate = librosa.load(save_path, sr=16000)

    # Process the audio sample
    input_features = processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt").input_features
    
    # Generate transcription
    with torch.no_grad():
        predicted_ids = model.generate(input_features)
        transcription = processor.decode(predicted_ids[0])
    transcription = transcription.replace("<|startoftranscript|><|notimestamps|>","")
    st.subheader("Audio Transcription")
    st.write(transcription)
    
    print(audio_array)
    pipe = pipeline(task="automatic-speech-recognition", model=model_path)
    st.subheader("FineTuned ASR")
    finetune_text=pipe(audio_array) 
    st.write(finetune_text['text'])
    