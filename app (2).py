import streamlit as st
import joblib
import base64
import re
import string

# Set page configuration
st.set_page_config(page_title="IMDB Sentiment Analyzer 🎬", layout="centered")

# Function to set the background image from a local file (with base64 encoding)
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

# Preprocessing functions (same as used during training)
def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def clean_text(text):
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def deep_clean(text):
    text = remove_html_tags(text)
    text = clean_text(text)
    return text

# Set background image
set_background("background_image.jpg")

# Load the trained model, vectorizer, and label encoder
model = joblib.load('logistic_regression_modelX.pkl')
vectorizer = joblib.load('tfidf_vectorizerX.pkl')
label_encoder = joblib.load('label_encoderX.pkl')  # <-- New addition

# Custom CSS
st.markdown(
    """
    <style>
    .stApp {{
        background-color: rgba(255, 255, 255, 0.8);
        padding: 2rem;
        border-radius: 15px;
    }}
    .stTextInput > div > div > input {{
        font-size: 16px;
    }}
    .stTextArea > div > div > textarea {{
        font-size: 16px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# App Header
st.title("🎬 IMDB Movie Review Sentiment Analyzer")
st.write("Welcome! Enter a movie review below and let the model predict if it's *Positive, Neutral, or Negative*! 💬")

# Text input
user_input = st.text_area("📝 Write your review here:")

# Predict button
if st.button("🔍 Analyze Sentiment"):
    if user_input.strip():
        cleaned_input = deep_clean(user_input)
        review_vector = vectorizer.transform([cleaned_input])
        prediction = model.predict(review_vector)[0]
        predicted_sentiment = label_encoder.inverse_transform([prediction])[0]  # <-- Correct prediction decoding

        # Display result
        st.subheader("🎯 Prediction Result:")
        st.success(f"✅ The review is predicted to be: *{predicted_sentiment.upper()}*")
    else:
        st.warning("⚠ Please enter a valid review before clicking Analyze.")

# Footer
st.markdown("---")
st.caption("Made with ❤ using Streamlit and Logistic Regression.")
