import streamlit as st
import joblib
import os
import gdown

MODEL_FILE = "logistic_regression_modelF.pkl"
VECTORIZER_FILE = "tfidf_vectorizer.pkl"

# Replace with your actual file IDs from Google Drive
model_file_id = "12DfR9Kf0pdJEPSx4yqWUA35aWQA7KCP9"
vectorizer_file_id = "tfidf_vectorizer (1).pkl"

# Function to download files from Google Drive
def download_from_gdrive(file_id, output_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)

# Download model if not exists
if not os.path.exists(MODEL_FILE):
    download_from_gdrive(model_file_id, MODEL_FILE)

if not os.path.exists(VECTORIZER_FILE):
    download_from_gdrive(vectorizer_file_id, VECTORIZER_FILE)

# Load model and vectorizer
model = joblib.load(MODEL_FILE)
vectorizer = joblib.load(VECTORIZER_FILE)

# App
st.title("🎬 IMDB Movie Review Sentiment Classifier (Google Drive Model)")
user_input = st.text_area("Write your review here:", height=150)

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a review.")
    else:
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)[0]
        st.success(f"✅ Predicted Sentiment: **{prediction.capitalize()}**")
