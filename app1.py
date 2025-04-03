import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model_path = 'email_spam.pkl'
model = joblib.load(model_path)

# Load the email dataset
data_path = '/mnt/data/mail_data.csv'
data = pd.read_csv(data_path)

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

# Upload CSV for batch classification
uploaded_file = st.file_uploader("Upload a CSV file for batch processing", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if 'email' in df.columns:
        df['Prediction'] = model.predict(df['email'])
        df['Prediction'] = df['Prediction'].apply(lambda x: "Spam" if x == 1 else "Not Spam")
        st.write(df)
    else:
        st.write("CSV file must contain a column named 'email'")
