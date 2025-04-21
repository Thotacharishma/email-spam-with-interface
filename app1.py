import streamlit as st
import joblib

# Load trained model
model = joblib.load("email_spam.pkl")

st.title("📧 Spam Mail Detector")

# User input
email_text = st.text_area("Enter your email content:")

if st.button("Predict"):
    if email_text.strip() == "":
        st.error("Please enter an email before predicting.")
    else:
        st.write(f"🔍 **Input Received:** {email_text}")  # Debugging line
        prediction = model.predict([email_text])[0]
        label = "Spam" if prediction == 1 else "Ham"
        st.success(f"📩 **Predicted Label:** {label}")
