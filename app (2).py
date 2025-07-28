import streamlit as st
import joblib
import os
import gdown

# === Filenames and Google Drive File ID ===
MODEL_FILE = "logistic_regression_modelF.pkl"
MODEL_FILE_ID = "12DfR9Kf0pdJEPSx4yqWUA35aWQA7KCP9"  # From your shared link

# === Download model from Google Drive if not already present ===
def download_model_from_drive(file_id, output_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)

if not os.path.exists(MODEL_FILE):
    st.info("📦 Downloading large model file from Google Drive...")
    download_model_from_drive(MODEL_FILE_ID, MODEL_FILE)

# === Load model and vectorizer ===
model = joblib.load(MODEL_FILE)
vectorizer = joblib.load("tfidf_vectorizer.pkl")  # Your local file

# === Streamlit App UI ===
st.title("🎬 IMDB Movie Review Sentiment Classifier")
st.markdown("Enter a movie review below and the model will classify it as **Positive**, **Neutral**, or **Negative**.")

# === Text input ===
user_input = st.text_area("Write your review here:", height=150)

# === Prediction ===
if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a review.")
    else:
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)[0]
        st.success(f"✅ Predicted Sentiment: **{prediction.capitalize()}**")
