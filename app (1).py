import streamlit as st
import joblib
import base64

# Page setup
st.set_page_config(page_title="🎬 Movie Sentiment Analyzer", page_icon="🎬", layout="centered")

# Background image setup
def set_background(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
        custom_css = f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');

        html, body, .stApp {{
            background: url("data:image/png;base64,{encoded}") no-repeat center center fixed;
            background-size: cover;
            font-family: 'Montserrat', sans-serif;
            color: #FFFFFF;
            text-align: center;
        }}

        h1 {{
            font-weight: 700;
            margin-top: 5vh;
            margin-bottom: 2rem;
            text-shadow: 2px 2px 10px rgba(0,0,0,0.6);
        }}

        .stTextArea textarea {{
            border: 2px solid #00ffc6;
            border-radius: 8px;
            background: rgba(0, 0, 0, 0.6);
            color: #ffffff;
            padding: 1rem;
            font-size: 1rem;
        }}

        .stButton>button {{
            background: linear-gradient(135deg, #00ffc6, #007cf0);
            border: none;
            color: white;
            border-radius: 30px;
            padding: 0.7em 2em;
            font-size: 16px;
            font-weight: bold;
            margin-top: 1rem;
            cursor: pointer;
            transition: all 0.3s ease;
        }}

        .stButton>button:hover {{
            transform: translateY(-3px) scale(1.05);
            background: white;
            color: #007cf0;
            border: 2px solid #00ffc6;
        }}

        .result-box {{
            background: rgba(0, 0, 0, 0.75);
            padding: 1rem 2rem;
            border-radius: 15px;
            margin-top: 1.5rem;
            display: inline-block;
            font-size: 1.3rem;
            font-weight: 600;
            color: #00ffc6;
        }}

        .stMarkdown p, .stAlert {{
            color: #ffffff;
        }}
        </style>
        """
    st.markdown(custom_css, unsafe_allow_html=True)

# Set your new background image
set_background("background_image.jpg") 

# Load model and vectorizer
model = joblib.load('logistic_regression_modelF.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# App content
st.title('🎬 Movie Sentiment Analyzer')

review = st.text_area('📝 Enter your movie review:')

if st.button('Predict Sentiment'):
    if review.strip():
        transformed_review = vectorizer.transform([review])
        prediction = model.predict(transformed_review)[0]

        sentiment = ['😡 Negative', '😐 Neutral', '😊 Positive']
        result_html = f'<div class="result-box">Result: {sentiment[prediction]}</div>'
        st.markdown(result_html, unsafe_allow_html=True)
    else:
        st.warning('⚠️ Please enter a valid review to analyze.')
