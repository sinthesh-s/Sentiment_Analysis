import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load('logistic_regression_modelF.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

st.title('Movie Review Sentiment Analysis')

review = st.text_area('Enter a movie review:')

if st.button('Predict'):
    if review.strip():
        # Transform input and make prediction
        transformed_review = vectorizer.transform([review])
        prediction = model.predict(transformed_review)[0]

        # Convert numeric label to text (edit according to your label order)
        if prediction == 2:
            sentiment = 'Positive'
        elif prediction == 0:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'

        st.success(f'Sentiment: **{sentiment}**')
    else:
        st.warning('Please enter a valid review.')
