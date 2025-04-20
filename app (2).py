import streamlit as st
import joblib
import numpy as np
import json
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# Load tokenizer
with open('tokenizer.json', 'r') as f:
    tokenizer_data = json.load(f)
    tokenizer = tokenizer_from_json(tokenizer_data)

# Load the trained BiLSTM model
model = load_model('bilstm_model.h5')

# Load Label Encoder
le = joblib.load('label_encoder.pkl')

# Set max length used during training
MAXLEN = 200

# Streamlit app layout
st.set_page_config(page_title="Sentiment Analysis", layout="centered")
st.title("🎬 IMDB MovieEEE Review Sentiment Analysis")
st.write("Enter a movie review below and the model will predict its sentiment (Positive, Negative, Neutral).")

# Input field
user_input = st.text_area("Enter your movie review:")

# Predict button
if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a valid review.")
    else:
        # Preprocess input
        sequence = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(sequence, maxlen=MAXLEN)

        # Predict
        probs = model.predict(padded)
        predicted_class = np.argmax(probs, axis=1)[0]
        predicted_label = le.inverse_transform([predicted_class])[0]
        confidence = np.max(probs)

        # Output
        st.success(f"Predicted Sentiment: **{predicted_label.capitalize()}**")
        st.info(f"Model Confidence: {confidence:.2%}")
