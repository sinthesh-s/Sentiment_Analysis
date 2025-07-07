import streamlit as st
import numpy as np
import joblib
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Constants
MAX_LEN = 200  # Must match the length used during training

# Load tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load label encoder
label_encoder = joblib.load('label_encoderX.pkl')

# Load model
model = load_model('sentiment_model.h5')

# App title
st.title("🎬 IMDB Movie Review Sentiment Classifier (BiLSTM)")
st.markdown("Enter a movie review below and the model will classify it as **Positive**, **Neutral**, or **Negative**.")

# User input
user_input = st.text_area("Write your review here:", height=150)

# Predict button
if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a review.")
    else:
        # Tokenize and pad the input
        sequence = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(sequence, maxlen=MAX_LEN)

        # Predict
        prediction_probs = model.predict(padded)
        predicted_class = np.argmax(prediction_probs, axis=1)[0]
        sentiment = label_encoder.inverse_transform([predicted_class])[0]

        # Display prediction
        st.success(f"✅ Predicted Sentiment: **{sentiment.capitalize()}**")
