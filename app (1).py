import streamlit as st
import joblib
import base64

st.set_page_config(page_title="Sentiment Analysis", layout="centered")

def set_background(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
        st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;700&display=swap');

        .stApp {{
            background: url("data:image/png;base64,{encoded}") no-repeat center center fixed;
            background-size: cover;
            font-family: 'Poppins', sans-serif;
        }}

        .glass-card {{
            background: rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 40px 30px;
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);
            max-width: 600px;
            margin: 100px auto;
            text-align: center;
            animation: fadeIn 1s ease forwards;
            border: 1px solid rgba(255,255,255,0.2);
        }}

        h1 {{
            font-size: 2.5rem;
            color: #ffffff;
            margin-bottom: 25px;
            text-shadow: 0 4px 10px rgba(0,0,0,0.6);
        }}

        label {{
            font-size: 1rem;
            color: #dddddd;
            margin-bottom: 0.5rem;
            display: block;
            text-align: left;
        }}

        .stTextArea textarea {{
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.3);
            border-radius: 12px;
            color: #fff;
            font-size: 15px;
            padding: 14px;
        }}

        .stButton>button {{
            background: linear-gradient(90deg, #ff4b4b, #ff7777);
            color: white;
            border: none;
            border-radius: 14px;
            padding: 0.7em 2em;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: 0.3s ease;
            margin-top: 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }}

        .stButton>button:hover {{
            background: white;
            color: #ff4b4b;
            transform: translateY(-2px) scale(1.03);
            border: 2px solid #ff4b4b;
        }}

        .stMarkdown p {{
            color: #eeeeee;
            font-size: 16px;
        }}

        .stAlert {{
            border-radius: 12px;
        }}

        @keyframes fadeIn {{
            0% {{ opacity: 0; transform: translateY(-20px); }}
            100% {{ opacity: 1; transform: translateY(0); }}
        }}
        </style>
        """, unsafe_allow_html=True)

set_background("background_image.jpg")

# Load model and vectorizer
model = joblib.load('logistic_regression_modelF.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

st.markdown('<div class="glass-card">', unsafe_allow_html=True)

st.title('🎬 Sentiment Analysis')

review = st.text_area('📝 Enter a movie review:')

if st.button('Predict'):
    if review.strip():
        transformed_review = vectorizer.transform([review])
        prediction = model.predict(transformed_review)[0]
        sentiment = ['Negative', 'Neutral', 'Positive']
        color = ['#FF4B4B', '#FFC107', '#00C851']

        st.markdown(
            f"<h3 style='color:{color[prediction]}; font-weight:700;'>Sentiment: {sentiment[prediction]}</h3>",
            unsafe_allow_html=True
        )
    else:
        st.warning('⚠️ Please enter a valid review.')

st.markdown('</div>', unsafe_allow_html=True)
