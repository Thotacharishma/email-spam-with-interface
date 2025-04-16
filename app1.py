import streamlit as st
import joblib
import os

# Load the trained model
model_path = 'email_spam.pkl'

if os.path.exists(model_path):
    model = joblib.load(model_path)
    st.success("Model loaded successfully.")
else:
    st.error(f"Model file not found at {model_path}. Please upload it.")
    st.stop()

# Streamlit UI
st.title("📧 Email Spam Classifier")
st.write("Enter your email content below and classify it as Spam or Not Spam.")

# Email input
email_text = st.text_area("✍️ Enter Email Text:", height=200)

# Prediction
if st.button("🚀 Classify Email"):
    if email_text.strip():
        try:
            prediction = model.predict([email_text])  # Pass as list for model
            result = "Spam" if prediction[0] == 1 else "Not Spam"
            st.success(f"🧠 Classification Result: **{result}**")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.warning("Please enter some text before classifying.")
