# app.py

import streamlit as st
import joblib
import base64
import re

# Set page configuration FIRST
st.set_page_config(page_title="IMDB Sentiment Analyzer 🎬", layout="centered")

# Background Image Setup
def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-attachment: fixed;
    }}
    .stApp .stTextInput > div > div > input {{
        font-size: 16px;
    }}
    .stApp .stTextArea > div > div > textarea {{
        font-size: 16px;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Set background image from local file
set_background("background_image.jpg")

# Load model, vectorizer, and label encoder
model = joblib.load('logistic_regression_modelZ.pkl')
vectorizer = joblib.load('tfidf_vectorizerZ.pkl')
label_encoder = joblib.load('label_encoderZ.pkl')

# Negation handler (must match training logic!)
def handle_negation(text):
    return re.sub(r'\bnot\s+(\w+)', r'not_\1', text.lower())

# Mapping labels back
label_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}

# App UI
st.title("🎬 IMDB Movie Review Sentiment Analyzer")
st.write("Welcome! Enter a movie review below and let the model predict if it's **Positive**, **Neutral**, or **Negative**! 💬")

user_input = st.text_area("📝 Write your review here:")

if st.button("🔍 Analyze Sentiment"):
    if user_input.strip():
        # Apply negation handling
        processed_input = handle_negation(user_input)

        # Vectorize input
        review_vector = vectorizer.transform([processed_input])

        # Predict
        prediction = model.predict(review_vector)[0]
        predicted_sentiment = label_mapping.get(prediction, "Unknown")

        st.subheader("🎯 Prediction Result:")
        st.success(f"✅ The review is predicted to be: **{predicted_sentiment.upper()}**")
    else:
        st.warning("⚠️ Please enter a valid review before clicking Analyze.")

st.markdown("---")
st.caption("Made with ❤️ using Streamlit and Logistic Regression.")
