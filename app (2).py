code = """
import streamlit as st
import joblib

# Load the model and vectorizer
model = joblib.load('lstm_sentiment_model__.keras')
vectorizer = joblib.load('tokenizer__.pkl')

st.title('Movie Review Sentiment Analysis')
review = st.text_area('Enter a movie review:')

if st.button('Predict'):
    if review.strip():
        transformed_review = vectorizer.transform([review])
        prediction = model.predict(transformed_review)[0]
        sentiment = 'Positive' if prediction == 'positive' else 'Negative'
        st.write(f'The sentiment is: *{sentiment}*')
    else:
        st.write('Please enter a valid review.')
"""

# Save the code to app.py
with open('app.py', 'w') as f:
    f.write(code)
print("app.py1 saved successfully.")
