import streamlit as st
import joblib
import pandas as pd
from gensim.parsing.preprocessing import remove_stopwords
import gensim

# Load the saved XGBoost model and vectorizer
model = joblib.load('xgb_news_classifier_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Preprocessing function using Gensim
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):  # Tokenization
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(token)
    return result

# Streamlit app
st.title("Fake News Classifier")

st.write("Enter the news title and text to classify it as real or fake.")

# Input text from the user
title = st.text_input("Title")
text = st.text_area("Text")

# Combine the title and text
if st.button("Classify"):
    if title and text:
        original_text = title + ' ' + text
        clean_text = preprocess(original_text)
        clean_joined = ' '.join(clean_text)

        # Transform the input text using the TF-IDF vectorizer
        text_vector = vectorizer.transform([clean_joined])

        # Predict the label
        prediction = model.predict(text_vector)[0]

        # Display the result
        if prediction == 1:
            st.success("This news is Real.")
        else:
            st.error("This news is Fake.")
    else:
        st.warning("Please enter both title and text.")
