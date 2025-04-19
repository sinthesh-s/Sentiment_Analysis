import streamlit as st
import joblib
import base64

# Page setup
st.set_page_config(page_title="🎬 Movie Sentiment Analyzer", page_icon="🎬", layout="centered")

# Modern clean background
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
        }}

        .glass {{
            background: rgba(255, 255, 255, 0.10);
            border-radius: 20px;
            padding: 2rem;
            max-width: 600px;
            margin: 5vh auto;
            backdrop-filter: blur(18px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.37);
            animation: fadeIn 1s ease;
            text-align: center;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        h1 {{
            color: #ffffff;
            font-weight: 700;
            text-shadow: 2px 2px 8px rgba(0,0,0,0.6);
            margin-bottom: 1rem;
        }}

        .stTextArea textarea {{
            background: rgba(0, 0, 0, 0.4);
            border: none;
            border-radius: 10px;
            color: #fff;
            padding: 1rem;
            font-size: 1rem;
            transition: 0.3s ease;
        }}

        .stTextArea textarea:focus {{
            outline: none;
            border: 2px solid #00ffc6;
            background: rgba(255, 255, 255, 0.15);
        }}

        .stButton>button {{
            background: linear-gradient(135deg, #00ffc6, #007cf0);
            border: none;
            color: white;
            border-radius: 30px;
            padding: 0.7em 2em;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
        }}

        .stButton>button:hover {{
            transform: translateY(-3px) scale(1.05);
            background: white;
            color: #007cf0;
            border: 2px solid #00ffc6;
        }}

        .stMarkdown p, .stAlert {{
            color: #ffffff;
        }}
        </style>
        """
    st.markdown(custom_css, unsafe_allow_html=True)

# Set your background image
set_background("background_image.png")

# Load your model and vectorizer
model = joblib.load('logistic_regression_modelF.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Glass box content
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.title('🎬 Movie Sentiment Analyzer')

review = st.text_area('📝 Enter your movie review here:')

if st.button('Predict'):
    if review.strip():
        transformed_review = vectorizer.transform([review])
        prediction = model.predict(transformed_review)[0]

        sentiment = ['😡 Negative', '😐 Neutral', '😊 Positive']
        st.success(f"Result: **{sentiment[prediction]}**")
    else:
        st.warning('⚠️ Please enter a valid review to analyze.')

st.markdown('</div>', unsafe_allow_html=True)
