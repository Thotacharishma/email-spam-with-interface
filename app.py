import streamlit as st
import joblib

# Load the model and vectorizer
model = joblib.load("email_spam.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("📧 Spam Mail Detector")

# User input
email_text = st.text_area("Enter your email content:")

if st.button("Predict"):
    if email_text.strip() == "":
        st.error("Please enter an email before predicting.")
    else:
        st.write(f"🔍 **Input Received:** {email_text}")
        
        transformed_text = vectorizer.transform([email_text])
        prediction = model.predict(transformed_text)[0]
        
        label = "Spam" if prediction == 1 else "Ham"
        st.success(f"📩 **Predicted Label:** {label}")
