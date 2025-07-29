import streamlit as st
import pickle
import gzip
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import base64

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
st.markdown("<p style='text-align:center; color:white;'>Enter a review below to detect its sentiment using LSTM model</p>", unsafe_allow_html=True)

review = st.text_area("Write your review here:", height=150)

if st.button("Predict Sentiment"):
    if review.strip():
        seq = tokenizer.texts_to_sequences([review])
        padded = pad_sequences(seq, maxlen=200)

        prediction = model.predict(padded)[0]
        label = prediction.argmax()

        sentiment = {0: "Negative", 1: "Neutral", 2: "Positive"}.get(label, "Unknown")
        st.markdown(f"<h3 style='color:white;'>Sentiment: <span style='color:#FFD700;'>{sentiment}</span></h3>", unsafe_allow_html=True)
    else:
        st.warning("Please enter a valid review.")
