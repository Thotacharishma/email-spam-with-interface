import streamlit as st
import joblib
import os

# Load model and vectorizer
model_path = 'email_spam.pkl'
vectorizer_path = 'vectorizer.pkl'

if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    st.success("Model and vectorizer loaded successfully.")
else:
    st.error("Model or vectorizer file not found. Please make sure both are in the same folder.")
    st.stop()

# Streamlit UI
st.title("📧 Email Spam Classifier")
st.write("Enter your email content below and classify it as Spam or Not Spam.")

email_text = st.text_area("✍️ Enter Email Text:", height=200)

if st.button("🚀 Classify Email"):
    if email_text.strip():
        try:
            # Vectorize the input first
            email_vector = vectorizer.transform([email_text])
            prediction = model.predict(email_vector)

            result = "Spam" if prediction[0] == 1 else "Not Spam"
            st.success(f"🧠 Classification Result: **{result}**")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.warning("Please enter some text before classifying.")
