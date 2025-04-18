import streamlit as st
import joblib
import base64

# Set page configuration
st.set_page_config(page_title="Sentiment Analysis", layout="centered")

# Function to set background image
def set_background(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
        background_image = f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """
        st.markdown(background_image, unsafe_allow_html=True)

set_background("background_image.jpg")

# Load your model and vectorizer
model = joblib.load('logistic_regression_modelF.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# App content
st.title('Sentiment Analysis App')
review = st.text_area('Enter a movie review:')

if st.button('Predict'):
    if review.strip():
        transformed_review = vectorizer.transform([review])
        prediction = model.predict(transformed_review)[0]

        sentiment = ['Negative', 'Neutral', 'Positive']
        st.success(f"Sentiment: **{sentiment[prediction]}**")
    else:
        st.warning('Please enter a valid review.')
