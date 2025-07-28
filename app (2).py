import streamlit as st
import joblib
import numpy as np

# Load TF-IDF vectorizer
with open("tfidf_vectorizer.pkl", "rb") as tfidf_file:
    vectorizer = joblib.load(tfidf_file)

# Load Logistic Regression model
with open("logistic_regression_modelF.pkl", "rb") as model_file:
    model = joblib.load(model_file)

# App title
st.title("🎬 IMDB Movie Review Sentiment Classifier (Logistic Regression)")
st.markdown("Enter a movie review below and the model will classify it as **Positive**, **Neutral**, or **Negative**.")

# User input
user_input = st.text_area("Write your review here:", height=150)

# Predict button
if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a review.")
    else:
        # Vectorize input
        transformed_input = vectorizer.transform([user_input])

        # Predict sentiment
        prediction = model.predict(transformed_input)[0]

        # Show result
        st.success(f"✅ Predicted Sentiment: **{prediction.capitalize()}**")
