import streamlit as st
import numpy as np
import joblib
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# Load Tokenizer from JSON
with open('tokenizer.json', 'r') as f:
    tokenizer_json = f.read()
    tokenizer = tokenizer_from_json(tokenizer_json)

# Load Label Encoder (Updated filename)
label_encoder = joblib.load('label_encoderX.pkl')

# Load Model
model = load_model('bilstm_model.h5')

# Constants
MAX_LEN = 200  # Must match the length used in training

# App UI
st.title("🎬 IMDB MoviEEEEe Review Sentiment Classifier (BiLSTM)")
st.markdown("Enter a movie review below and the model will classify it as Positive, Neutral, or Negative.")

# User Input
user_input = st.text_area("Write your review here:", height=150)

# Predict Button
if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        # Preprocess
        sequence = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(sequence, maxlen=MAX_LEN)

        # Predict
        prediction_probs = model.predict(padded)
        predicted_class = np.argmax(prediction_probs, axis=1)[0]
        sentiment = label_encoder.inverse_transform([predicted_class])[0]

        # Display
        st.success(f"Predicted Sentiment: **{sentiment.capitalize()}**")
