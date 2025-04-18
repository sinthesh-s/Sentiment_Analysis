import streamlit as st
import joblib
import base64

# Set page configuration
st.set_page_config(page_title="Sentiment Analysis", layout="centered")

# Function to set stylish background with overlay
def set_background(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
        background_style = f"""
        <style>
        .stApp {{
            background: url("data:image/png;base64,{encoded}") no-repeat center center fixed;
            background-size: cover;
        }}

        /* Full screen dark overlay */
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.65);
            z-index: -1;
        }}

        /* Centered glassmorphism card */
        .main-container {{
            background: rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 30px;
            max-width: 600px;
            margin: 100px auto;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            color: white;
        }}

        .stTextArea textarea {{
            background-color: rgba(0,0,0,0.6);
            color: white;
            border-radius: 8px;
        }}

        .stButton>button {{
            background-color: #ff4b4b;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5em 1em;
            font-weight: bold;
            transition: 0.3s ease;
        }}

        .stButton>button:hover {{
            background-color: white;
            color: #ff4b4b;
            border: 1px solid #ff4b4b;
        }}

        h1, .stMarkdown, .stAlert {{
            color: white;
        }}
        </style>
        """
        st.markdown(background_style, unsafe_allow_html=True)

# Apply background and styling
set_background("background_image.jpg")  # Make sure this is the correct filename

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
