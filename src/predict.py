import joblib
import os
from preprocess import clean_text

# Load model and vectorizer once (important for performance)
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Ensure output folder exists
os.makedirs("outputs", exist_ok=True)


# ==============================
# REUSABLE FUNCTION (CORE PART)
# ==============================
def predict_sentiment(text):
    """
    Takes raw text input and returns sentiment prediction.
    """

    try:
        # Step 1: Clean input text
        cleaned_text = clean_text(text)

        # Step 2: Convert text into TF-IDF vector
        vector = vectorizer.transform([cleaned_text])

        # Step 3: Predict sentiment
        prediction = model.predict(vector)[0]

        # Step 4: Logging (safe & correct)
        with open("outputs/prediction_logs.txt", "a", encoding="utf-8") as f:
            f.write(f"{text} → {prediction}\n")

        return prediction

    except Exception as e:
        return f"Error: {str(e)}"


# ==============================
# TESTING BLOCK (OPTIONAL)
# ==============================
if __name__ == "__main__":
    print("Sentiment Prediction System Ready 🔥")

    while True:
        user_input = input("\nEnter text (or type 'exit'): ")

        if user_input.lower() == "exit":
            break

        result = predict_sentiment(user_input)
        print("Predicted Sentiment:", result)