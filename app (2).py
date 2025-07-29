import streamlit as st
import numpy as np
import os
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle

# Set page title
st.set_page_config(page_title="Sentiment Analysis", layout="centered")

# -----------------------
# Model and tokenizer paths
MODEL_PATH = "lstm_sentiment_model__.keras"
TOKENIZER_PATH = "tokenizer__.pkl"

# Load LSTM model
model = load_model(MODEL_PATH)

# Load tokenizer
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

# Define max sequence length (must match training)
MAX_LEN = 200

# Prediction function
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
