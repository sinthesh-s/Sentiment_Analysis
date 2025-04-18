import streamlit as st
import joblib
import base64

# Set page configuration
st.set_page_config(page_title="Sentiment Analysis", layout="centered")

# Function to set stylish background with animations
def set_background(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
        background_style = f"""
        <style>
        .stApp {{
            background: url("data:image/png;base64,{encoded}") no-repeat center center fixed;
            background-size: cover;
        }}

        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.6);
            z-index: -1;
        }}

        /* Animate fade-in and slide */
        @keyframes fadeSlide {{
            0% {{
                opacity: 0;
                transform: translateY(-20px);
            }}
            100% {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        .main-container {{
            background: rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 30px;
            max-width: 600px;
            margin: 100px auto;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            color: white;
            animation: fadeSlide 1s ease-out;
        }}

        .stTextArea textarea {{
            background-color: rgba(0,0,0,0.6);
            color: white;
            border-radius: 8px;
            border: 1px solid #ccc;
            transition: all 0.3s ease;
        }}

        .stTextArea textarea:focus {{
            transform: scale(1.02);
            border-color: #ff4b4b;
            box-shadow: 0 0 8px #ff4b4b44;
        }}

        .stButton>button {{
            background-color: #ff4b4b;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5em 1em;
            font-weight: bold;
            transition: 0.3s ease;
            animation: fadeSlide 1.5s ease-out;
        }}

        .stButton>button:hover {{
            background-color: white;
            color: #ff4b4b;
            border: 1px solid #ff4b4b;
            transform: scale(1.05);
        }}

        h1, .stMarkdown, .stAlert {{
            color: white;
        }}
        </style>
        """
        st.markdown(background_style, unsafe_allow_html=True)

# Apply animated background
set_background("background_image.jpg")

# Load model and vectorizer
model = joblib.load('logistic_regression_modelF.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Animated UI card
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
