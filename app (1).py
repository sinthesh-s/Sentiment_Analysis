import streamlit as st
import joblib
import base64

# Set page configuration
st.set_page_config(page_title="Sentiment Analysis", layout="centered")

# Function to set the background
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
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }}

        @keyframes fadeSlide {{
            0% {{ opacity: 0; transform: translateY(-20px); }}
            100% {{ opacity: 1; transform: translateY(0); }}
        }}

        .main-container {{
            background: rgba(255, 255, 255, 0.12);
            border-radius: 20px;
            padding: 35px 30px 30px 30px;
            width: 100%;
            max-width: 600px;
            backdrop-filter: blur(15px);
            box-shadow: 0 8px 32px rgba(0,0,0,0.5);
            animation: fadeSlide 1s ease-out;
            color: #ffffff;
            text-align: center;
        }}

        h1 {{
            color: #ffffff;
            text-shadow: 2px 2px 8px rgba(0,0,0,0.7);
            font-size: 2.3rem;
            margin-bottom: 20px;
        }}

        .stTextArea textarea {{
            background: rgba(0,0,0,0.3);
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
            background: rgba(255, 255, 255, 0.15);
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
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
            margin-top: 15px;
        }}

        .stButton>button:hover {{
            background: white;
            color: #ff4b4b;
            border: 2px solid #ff4b4b;
            transform: translateY(-2px) scale(1.05);
        }}

        .stMarkdown p, .stAlert {{
            color: #f0f0f0;
        }}
        </style>
        """
    st.markdown(background_style, unsafe_allow_html=True)

# Apply background
set_background("background_image.jpg")  # Make sure your file name matches!

# Load model and vectorizer
model = joblib.load('logistic_regression_modelF.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Start of the glassmorphism container
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

st.markdown('</div>', unsafe_allow_html=True)  # Close glass container
