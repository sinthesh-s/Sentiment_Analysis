import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('cleaned_dataset_134.csv')
# Load the saved model and vectorizer
model = joblib.load('logistic_regression_modelF.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Set the title and header
st.title('Sentiment Analysis App')
st.subheader('Predict sentiment from user reviews!')

# Add custom CSS to set the image as the background
st.markdown("""
    <style>
        .stApp {
            background-image: url('background_image.jpg');
            background-size: cover;
            background-position: center center;
            background-attachment: fixed;
            color: white;
        }
        .block-container {
            padding-top: 50px;
        }
        .stTextInput input {
            background-color: rgba(255, 255, 255, 0.7);
        }
    </style>
    """, unsafe_allow_html=True)

# User input for sentiment analysis
user_review = st.text_input("Enter a review:")

if user_review:
    # Vectorize the input text
    review_vectorized = vectorizer.transform([user_review])

    # Predict sentiment
    prediction = model.predict(review_vectorized)
    
    # Map prediction to sentiment
    sentiment = ['Negative', 'Neutral', 'Positive']
    st.write(f"Prediction: {sentiment[prediction[0]]}")

    # Add some feedback based on the prediction
    if prediction[0] == 0:
        st.write("It seems like this review is Negative. Let's work on improving it!")
    elif prediction[0] == 1:
        st.write("This review is Neutral. There's a bit of both positive and negative aspects.")
    else:
        st.write("This review is Positive. Great feedback!")

# Add an expander with more information
with st.expander("About the model"):
    st.write("""
        This model uses natural language processing (NLP) to classify text reviews as either negative, neutral, or positive.
        The model was trained using TF-IDF vectorization and a logistic regression classifier.
    """)

# Display a sample chart (for example, a bar chart of sentiment counts from a dataset)
if st.checkbox("Show sentiment distribution"):
    fig = plt.figure(figsize=(8, 6))
    sns.countplot(x="sentiment", data=df)
    st.pyplot(fig)
