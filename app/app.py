import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import joblib
import pandas as pd
import plotly.express as px

from src.preprocess import clean_text

# Load model and vectorizer
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Page config
st.set_page_config(page_title="Sentiment AI Dashboard", page_icon="📊", layout="wide")

st.title("📊 Social Media Sentiment Analysis Dashboard")
st.markdown("<h1 style='text-align: center;'>Sentiment Analysis AI Dashboard</h1>", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Input Options")
option = st.sidebar.radio("Choose input type:", ["Single Text", "CSV File"])

# ---------------- SINGLE TEXT ----------------
if option == "Single Text":
    st.subheader("Enter Text for Prediction")

    text = st.text_area("Type your comment here:")

    if st.button("Predict Sentiment"):
        if text.strip() != "":
            cleaned = clean_text(text)
            vector = vectorizer.transform([cleaned])
            prediction = model.predict(vector)[0]

            if prediction == "positive":
                st.success(f"😊 Positive Sentiment")
            elif prediction == "negative":
                st.error(f"😡 Negative Sentiment")
            else:
                st.info(f"😐 Neutral Sentiment")
        else:
            st.warning("Please enter some text!")

# ---------------- CSV UPLOAD ----------------
else:
    st.subheader("Upload CSV File")

    file = st.file_uploader("Upload file with a 'text' column", type=["csv"])

    if file is not None:
        df = pd.read_csv(file)

        if "text" not in df.columns:
            st.error("CSV must contain a 'text' column")
        else:
            df["cleaned"] = df["text"].apply(clean_text)
            X = vectorizer.transform(df["cleaned"])
            df["sentiment"] = model.predict(X)

            st.write("### Preview Data")
            st.dataframe(df.head())

            # Pie Chart
            fig1 = px.pie(df, names="sentiment", title="Sentiment Distribution")
            st.plotly_chart(fig1)

            # Bar Chart
            fig2 = px.histogram(df, x="sentiment", color="sentiment")
            st.plotly_chart(fig2)

            # Download result
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results", csv, "sentiment_results.csv", "text/csv")