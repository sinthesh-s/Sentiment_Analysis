import streamlit as st
import numpy as np
import gdown
import os
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Set page title
st.set_page_config(page_title="Sentiment Analysis", layout="centered")

# -----------------------
# Download model if not exists
MODEL_PATH = "lstm_sentiment_model.h5"
if not os.path.exists(MODEL_PATH):
    file_id = "1PtnS4mQ5Es3Qjl1jfMSf5a5Lai420Ppi"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, MODEL_PATH, quiet=False)

# -----------------------
# Load LSTM model
model = load_model(MODEL_PATH)

# Load tokenizer
import pickle
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# -----------------------
# Define max length (must match training)
MAX_LEN = 200

# Predict function
def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    pred = model.predict(padded)[0]
    sentiment = np.argmax(pred)
    label = ["Negative", "Neutral", "Positive"][sentiment]
    confidence = float(pred[sentiment])
    return label, confidence

# -----------------------
# Streamlit UI
st.title("Sentiment Analysis App")
user_input = st.text_area("Enter your comment:", height=150)

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        label, confidence = predict_sentiment(user_input)
        st.markdown(f"**Prediction:** {label}")
        st.markdown(f"**Confidence:** {confidence:.2f}")
