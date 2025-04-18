import streamlit as st
import joblib
import base64

# Set page configuration
st.set_page_config(page_title="Sentiment Analysis", layout="centered")

# Function to set stylish background
def set_background(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
        background_style = f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;600&display=swap');

        .stApp {{
            background: url("data:image/png;base64,{encoded}") no-repeat center center fixed;
            background-size: cover;
            font-family: 'Poppins', sans-serif;
        }}

        @keyframes fadeSlide {{
            0% {{ opacity: 0; transform: translateY(-30px); }}
            100% {{ opacity: 1; transform: translateY(0); }}
        }}

        .main-container {{
            background: rgba(255, 255, 255, 0.08);
            border-radius: 20px;
            padding: 35px;
            max-width: 650px;
            margin: 100px auto;
            backdrop-filter: blur(12px);
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);
            animation: fadeSlide 1s ease-out;
            color: #ffffff;
        }}

        .stTextArea textarea {{
            background: rgba(255, 255, 255, 0.1);
            color: #f0f0f0;
            border: none;
            border-radius: 10px;
            padding: 12px;
            font-size: 15px;
            transition: all 0.3s ease;
        }}

        .stTextArea textarea:focus {{
            border: 2px solid #ff4b4b;
            outline: none;
            background: rgba(255, 255, 255, 0.2);
            color: white;
        }}

        .stButton>button {{
            background: linear-gradient(135deg, #ff4b4b, #ff7777);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.6em 1.5em;
            font-weight: 600;
            font-size: 16px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            transition: all 0.3s ease;
        }}

        .stButton>button:hover {{
            background: white;
            color: #ff4b4b;
            border: 2px solid #ff4b4b;
            transform: translateY(-2px) scale(1.04);
            box-shadow: 0 6px 25px rgba(0, 0, 0, 0.35);
        }}

        h1 {{
            color: #ffffff;
            text-align: center;
            text-shadow: 2px 2px 8px rgba(0,0,0,0.6);
            font-size: 2.5rem;
            margin-bottom: 25px;
        }}

        .stMarkdown p {{
            color: #eeeeee;
            font-size: 16px;
        }}
        </style>
        """
    st.markdown(background_style, unsafe_allow_html=True)

# Apply background
set_background("background_image.jpg")  # Make sure the filename is correct

# Load model and vectorizer
model = joblib.load('logistic_regression_modelF.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Layout inside a styled container
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.title('🎬 Sentiment Analysis App')

review = st.text_area('📝 Enter a movie review:')

if st.button('Predict'):
    if review.strip():
        transformed_review = vectorizer.transform([review])
        prediction = model.predict(transformed_review)[0]

        sentiment = ['Negative', 'Neutral', 'Positive']
        st.success(f"✅ Sentiment: **{sentiment[prediction]}**")
    else:
        st.warning('⚠️ Please enter a valid review.')

st.markdown('</div>', unsafe_allow_html=True)
