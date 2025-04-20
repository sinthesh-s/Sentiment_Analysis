import streamlit as st
import joblib

# Load the model and vectorizer
model = joblib.load('logistic_regression_modelK.pkl')
vectorizer = joblib.load('tfidf_vectorizerK.pkl')

st.title('🎬 Movie Review Sentiment Analysis')
review = st.text_area('📝 Enter a movie review:')

if st.button('Predict Sentiment'):
    if review.strip():
        transformed_review = vectorizer.transform([review])
        prediction = model.predict(transformed_review)[0]
        sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        sentiment = sentiment_map.get(prediction, "Unknown")
        st.success(f'The sentiment is: **{sentiment}**')
    else:
        st.warning('⚠️ Please enter a valid review before predicting!')
