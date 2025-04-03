import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model_path = 'email_spam.pkl'
model = joblib.load(model_path)



# Streamlit UI
st.title("Email Spam Classifier")
st.write("Upload an email dataset or enter text to classify.")

# Email input
email_text = st.text_area("Enter Email Text:")

if st.button("Classify Email"):
    if email_text.strip():
        prediction = model.predict([email_text])
        result = "Spam" if prediction[0] == 1 else "Not Spam"
        st.write(f"### Classification: {result}")
    else:
        st.write("Please enter some text to classify.")

