import streamlit as st
import joblib
import os
import gdown

# File paths and Google Drive file ID
MODEL_FILE = "logistic_regression_modelF.pkl"
VECTORIZER_FILE = "tfidf_vectorizer (1).pkl"
model_file_id = "12DfR9Kf0pdJEPSx4yqWUA35aWQA7KCP9"  # Google Drive ID for model

# Function to download from Google Drive
def download_from_gdrive(file_id, output_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)

# Download model file if not present
if not os.path.exists(MODEL_FILE):
    st.info("📦 Downloading model from Google Drive...")
    download_from_gdrive(model_file_id, MODEL_FILE)

# Load model and vectorizer
model = joblib.load(MODEL_FILE)
vectorizer = joblib.load(VECTORIZER_FILE)

# App UI
st.title("🎬 IMDB Movie Review Sentiment Classifier")
st.markdown("Enter a movie review below and the model will classify it as **Positive**, **Neutral**, or **Negative**.")

# Text input
user_input = st.text_area("Write your review here:", height=150)

# Predict button
if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a review.")
    else:
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)[0]
        st.success(f"✅ Predicted Sentiment: **{prediction.capitalize()}**")
