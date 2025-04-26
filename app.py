import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the vectorizer and model
vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("email_spam.pkl")

# Streamlit app layout
st.set_page_config(page_title="Email/SMS Spam Classifier", layout="centered")
st.title("📧 Email & SMS Spam Classifier")

st.markdown("Enter a message below to check if it's spam or not:")

# Text input
user_input = st.text_area("Message", height=150)

# Predict button
if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        # Vectorize and predict
        input_data = vectorizer.transform([user_input])
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data).max()

        # Display result
        if prediction == "spam":
            st.error(f"🚫 Spam detected with {prob*100:.2f}% confidence.")
        else:
            st.success(f"✅ Ham (Not Spam) with {prob*100:.2f}% confidence.")

# Optional: View some raw data from mail_data.csv
with st.expander("📂 View Sample Data"):
    try:
        df = pd.read_csv("mail_data.csv")
        st.dataframe(df.sample(5))
    except Exception as e:
        st.warning("Could not load mail_data.csv.")
