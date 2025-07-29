import streamlit as st
import pickle
import gzip
import os
import base64
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# ---- Set page layout ----
st.set_page_config(page_title="Sentiment Analysis", layout="centered")

# ---- Background Styling ----
def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    style = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .stTextInput > label, .stTextArea > label {{
        color: white;
        font-weight: bold;
    }}
    </style>
    """
    st.markdown(style, unsafe_allow_html=True)

# ---- Load Model and Tokenizer ----
model = load_model("lstm_sentiment_model.h5")

with gzip.open("tokenizer.pkl.gz", "rb") as f:
    tokenizer = pickle.load(f)

# ---- UI ----
set_background("background_image.jpg")

st.markdown("<h1 style='text-align:center; color:white;'>🎬 Movie Review Sentiment Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:white;'>Enter a review below to detect its sentiment using an LSTM model.</p>", unsafe_allow_html=True)

review = st.text_area("Write your review here:", height=150)

if st.button("Predict Sentiment"):
    if review.strip():
        # Tokenize and pad
        seq = tokenizer.texts_to_sequences([review])
        padded = pad_sequences(seq, maxlen=200)  # Make sure 200 matches training

        # Predict
        prediction = model.predict(padded)

        # Debug: Show raw output
        st.write("🔍 Raw Prediction Output:", prediction)

        # Try both possibilities:
        if prediction.shape[1] == 1:
            # Model outputs a class directly (e.g., [[1]])
            label = int(prediction[0][0])
        else:
            # Model outputs softmax probabilities (e.g., [[0.1, 0.7, 0.2]])
            label = np.argmax(prediction[0])

        sentiment = {0: "Negative", 1: "Neutral", 2: "Positive"}.get(label, "Unknown")
        st.markdown(f"<h3 style='color:white;'>Sentiment: <span style='color:#FFD700;'>{sentiment}</span></h3>", unsafe_allow_html=True)
    else:
        st.warning("Please enter a valid review.")
