# app.py

import streamlit as st
import joblib
import base64
import re
import numpy as np

# Set page configuration
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
        animation: fadeInBackground 2s ease;
    }}
    @keyframes fadeInBackground {{
        0% {{ opacity: 0; }}
        100% {{ opacity: 1; }}
    }}
    .stTextInput > div > div > input, 
    .stTextArea > div > div > textarea {{
        font-size: 16px;
        border-radius: 10px;
        padding: 0.5rem;
        background-color: rgba(0, 0, 0, 0.6);
        color: white !important;
    }}
    .stButton button {{
        background: linear-gradient(90deg, #007cf0, #00dfd8);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.8rem;
        font-weight: 700;
        font-size: 16px;
        transition: 0.3s ease-in-out;
        box-shadow: 0 4px 10px rgba(0, 124, 240, 0.4);
    }}
    .stButton button:hover {{
        background: linear-gradient(90deg, #005ecb, #00c4be);
        transform: scale(1.05);
        box-shadow: 0 6px 14px rgba(0, 124, 240, 0.6);
    }}
    .stSubheader, .stCaption, .stMarkdown, .stText, .stTitle, .stHeader {{
        color: white;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.7);
        font-weight: 600;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Set background image
set_background("background_image.jpg")

# Sidebar
st.sidebar.title("About 🎬")
st.sidebar.markdown("""
This app uses **Multinomial Logistic Regression** to predict the sentiment of IMDB movie reviews.

[View Source on GitHub](https://github.com/sinthesh-s/Sentiment_Analysis)
""")

# Load model, vectorizer, and label encoder
model = joblib.load('logistic_regression_modelZ.pkl')
vectorizer = joblib.load('tfidf_vectorizerZ.pkl')
label_encoder = joblib.load('label_encoderZ.pkl')

# Negation handler
def handle_negation(text):
    return re.sub(r'\bnot\s+(\w+)', r'not_\1', text.lower())

label_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}

# App UI
st.title("🎬 IMDB Movie Review Sentiment Analyzer")
st.write("💡 Enter your movie review and let AI predict the sentiment: **Positive**, **Neutral**, or **Negative**.")

user_input = st.text_area("📝 Write your review here:")

if st.button("🔍 Predict Sentiment"):
    if user_input.strip():
        with st.spinner('Analyzing sentiment... 🎬🧠'):
            processed_input = handle_negation(user_input)
            review_vector = vectorizer.transform([processed_input])

            prediction = model.predict(review_vector)[0]
            prediction_proba = model.predict_proba(review_vector)[0]

            predicted_sentiment = label_mapping.get(prediction, "Unknown")

            st.subheader("🎯 Prediction Result:")
            st.success(f"✅ The review is predicted to be: **{predicted_sentiment.upper()}**")

            st.markdown("### 📊 Prediction Confidence:")
            chart_data = {
                label_mapping[i]: [round(prob * 100, 2)]
                for i, prob in enumerate(prediction_proba)
            }
            st.bar_chart(chart_data)
    else:
        st.warning("⚠️ Please enter a valid review before clicking Predict.")

st.markdown("---")
st.caption("Made with ❤️ using Streamlit and Logistic Regression.")
