import streamlit as st
import joblib
import base64

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

# Set background image
set_background("background_image.jpg")

# Load the trained model and vectorizer
model = joblib.load('logistic_regression_modelK.pkl')
vectorizer = joblib.load('tfidf_vectorizerK.pkl')

# Sentiment label mapping (adjust if necessary)
label_mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

# Streamlit page config
st.set_page_config(page_title="IMDB Sentiment Analyzer 🎬", layout="centered")

# Custom CSS for font styling
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
st.write("Welcome! Enter a movie review below and let the model predict if it's **Positive**, **Neutral**, or **Negative**! 💬")

# Text input
user_input = st.text_area("📝 Write your review here:")

# Predict button
if st.button("🔍 Analyze Sentiment"):
    if user_input.strip():
        # Transform input using TF-IDF vectorizer
        review_vector = vectorizer.transform([user_input])

        # Predict sentiment
        prediction = model.predict(review_vector)[0]
        predicted_sentiment = label_mapping.get(prediction, "Unknown")

        # Display result
        st.subheader("🎯 Prediction Result:")
        st.success(f"✅ The review is predicted to be: **{predicted_sentiment.upper()}**")
    else:
        st.warning("⚠️ Please enter a valid review before clicking Analyze.")

# Footer
st.markdown("---")
st.caption("Made with ❤️ using Streamlit and Logistic Regression.")
