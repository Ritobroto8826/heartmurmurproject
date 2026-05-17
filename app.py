import streamlit as st
import numpy as np

from model.model_loader import load_model
from audio.preprocessing import load_audio, extract_mfcc
from ui.visualizations import plot_waveform
from utils.logger import setup_logger

logger = setup_logger("HeartMurmurApp")

# Load model
model = load_model()

# Class mapping
classes = {
    0: "Artifact",
    1: "Normal",
    2: "Murmur"
}

st.title("Heart Murmur Detection with LSTM")

uploaded_file = st.file_uploader(
    "Upload a heart sound recording (WAV/MP3 format)",
    type=["wav", "mp3"]
)

if uploaded_file is not None:
    try:
        # Load audio
        y, sr = load_audio(uploaded_file)

        # Show waveform
        st.subheader("Waveform of Input Sound")
        fig = plot_waveform(y, sr)
        st.pyplot(fig)

        # Extract MFCC features
        X_input = extract_mfcc(y, sr)

        # Model prediction
        prediction = model.predict(X_input)

        # Predicted class index
        predicted_index = np.argmax(prediction, axis=1)[0]

        # Map index to class label
        predicted_class = classes[predicted_index]

        # Display results
        st.subheader("Prediction Result")
        st.write(f"Predicted Class: **{predicted_class}**")

        # Optional: show probabilities clearly
        st.write("Prediction Probabilities:")
        st.write({
            "Artifact": float(prediction[0][0]),
            "Normal": float(prediction[0][1]),
            "Murmur": float(prediction[0][2])
        })

        logger.info(
            f"Prediction completed for uploaded file: {uploaded_file.name}"
        )

    except Exception as e:
        logger.exception("Error processing uploaded file")

        st.error(
            "❌ An error occurred while processing the audio file. "
            "Please ensure it's a valid WAV or MP3 file and try again."
        )
