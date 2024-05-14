import gradio as gr
import speech_recognition as sr
import joblib
import numpy as np
from scipy.io.wavfile import write
import tempfile
import os

# Load the model and vectorizer
model = joblib.load('/Users/chamodyaavishka/Desktop/ML projects/Hate-Speech_detection/model.joblib')
cv = joblib.load('/Users/chamodyaavishka/Desktop/ML projects/Hate-Speech_detection/vectorizer.joblib')

def recognize_speech(audio_data):
    recognizer = sr.Recognizer()
    sample_rate, audio_array = audio_data

    with tempfile.NamedTemporaryFile(delete=True, suffix='.wav') as tmpfile:
        wav_path = tmpfile.name
        write(wav_path, sample_rate, audio_array)
        with sr.AudioFile(wav_path) as source:
            audio_recorded = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_recorded)
            except sr.UnknownValueError:
                return ""  

    return text.lower()  # Assuming the model was trained with lowercased text

def predict_speech(audio_data):
    text = recognize_speech(audio_data)
    if text:  # Proceed if text was recognized
        text_vector = cv.transform([text])
        if text_vector.shape[1] != model.n_features_in_:
            print(f"Debug: Text = {text}")
            print(f"Debug: Generated {text_vector.shape[1]} features, expected {model.n_features_in_}.")
            return f"Feature mismatch: Model expects {model.n_features_in_} features, received {text_vector.shape[1]}"
        prediction = model.predict(text_vector)[0]
    else:
        prediction = "No speech detected or speech not recognized."

    return prediction

# Set up the Gradio interface
iface = gr.Interface(
    fn=predict_speech,
    inputs=gr.Audio(),
    outputs="text",
    title="Speech to Offensive Content Detector",
    description="Speak into the microphone and the model will predict if the speech is offensive.",
    allow_flagging="never"
)

# Launch the interface with additional debugging options
iface.launch(debug=True, share=True)  # Set share=True to create a public link
