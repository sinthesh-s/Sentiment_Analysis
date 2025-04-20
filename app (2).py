from tensorflow.keras.models import load_model
import joblib
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = load_model("bilstm_model.h5")
tokenizer = joblib.load("tokenizer.pkl")

# Prediction function
def predict_sentiment(text):
    # Clean and preprocess the text as during training
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=200)  # Use same MAXLEN from training
    prediction = model.predict(padded)
    predicted_class = (prediction > 0.5).astype("int32")[0][0]
    return "positive" if predicted_class == 1 else "negative"
