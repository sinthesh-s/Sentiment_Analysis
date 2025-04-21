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
        background-color: #ffffffcc;
    }}
    .stButton button {{
        background-color: #FF4B4B;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        font-weight: bold;
        transition: 0.4s ease;
        box-shadow: 0 4px 14px rgba(255, 75, 75, 0.4);
    }}
    .stButton button:hover {{
        background-color: #e63636;
        transform: translateY(-2px);
        box-shadow: 0 6px 22px rgba(255, 28, 28, 0.6);
    }}
    .stSubheader, .stCaption {{
        font-weight: 600;
        color: #fff;
        text-shadow: 0 1px 3px rgba(0,0,0,0.4);
    }}
    .main-container {{
        background-color: rgba(255, 255, 255, 0.9);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
        animation: fadeInCard 1.2s ease;
    }}
    @keyframes fadeInCard {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
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

# Main App Layout
st.markdown("<div class='main-container'>", unsafe_allow_html=True)

st.title("🎬 IMDB Movie Review Sentiment Analyzer")
st.write("💡 Enter your movie review and let AI predict the sentiment: **Positive**, **Neutral**, or **Negative**.")

user_input = st.text_area("📝 Write your review here:")

if st.button("🔍 Analyze Sentiment"):
    if user_input.strip():
        processed_input = handle_negation(user_input)
        review_vector = vectorizer.transform([processed_input])
        prediction = model.predict(review_vector)[0]
        predicted_sentiment = label_mapping.get(prediction, "Unknown")
        st.subheader("🎯 Prediction Result:")
        st.success(f"✅ The review is predicted to be: **{predicted_sentiment.upper()}**")
    else:
        st.warning("⚠️ Please enter a valid review before clicking Analyze.")

st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("Made with ❤️ using Streamlit and Logistic Regression.")
