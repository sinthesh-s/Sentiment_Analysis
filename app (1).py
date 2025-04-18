import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved model and vectorizer
model = joblib.load('logistic_regression_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Set the title and header with custom styling
st.markdown('<p class="title">Movie Review Sentiment Analysis</p>', unsafe_allow_html=True)
st.subheader('Predict sentiment from user reviews!')

# Add an image for visual appeal (ensure you have an image file)
st.image('your_image.png', caption='Sentiment Analysis', use_column_width=True)

# Add custom CSS for layout and styling
st.markdown("""
    <style>
        .title {
            font-size: 32px;
            color: #4CAF50;
            text-align: center;
            font-weight: bold;
        }
        .stApp {
            font-family: "Arial", sans-serif;
        }
        .streamlit-expanderHeader {
            font-size: 20px;
        }
        .main-container {
            background-color: #f4f4f9;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .button {
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
        }
        .button:hover {
            background-color: #45a049;
        }
        body {
            background-color: #f0f2f6;
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

# Add an expander with more information about the model
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

# Footer with GitHub link
st.markdown("### Created by Your Name | [GitHub](https://github.com/yourusername)")
