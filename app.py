import streamlit as st
import numpy as np
from model.model_loader import load_model
from audio.preprocessing import load_audio, extract_mfcc
from ui.visualizations import plot_waveform
from utils.logger import setup_logger

logger = setup_logger("HeartMurmurApp")

model = load_model()

st.title("Heart Murmur Detection with LSTM")

uploaded_file = st.file_uploader("Upload a heart sound recording (WAV format)", type=["wav", "mp3"])

if uploaded_file is not None:
    try:
        y, sr = load_audio(uploaded_file)

        st.subheader("Waveform of Input Sound")
        fig = plot_waveform(y, sr)
        st.pyplot(fig)

        X_input = extract_mfcc(y, sr)

        prediction = model.predict(X_input)
        predicted_class = np.argmax(prediction, axis=1)[0]

        classes = ["Artifact", "Murmur", "Normal"]

        # Get predicted class index
        predicted_index = np.argmax(prediction, axis=1)[0]

        # Get predicted class name
        predicted_class = classes[predicted_index]

        st.subheader("Prediction Result")
        st.write(f"Predicted Class: **{predicted_class}**")
        st.write("Raw Prediction Probabilities:", prediction)

        logger.info(f"Prediction made for uploaded file: {uploaded_file.name} completed")

    except Exception as e:
        logger.exception("Error processing uploaded file")
        st.error("❌ An error occurred while processing the audio file. Please ensure it's a valid WAV or MP3 file and try again.")