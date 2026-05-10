import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load data
df = pd.read_csv("data/cleaned_data.csv")

# Load model
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

X = vectorizer.transform(df['cleaned_text'])
y = df['sentiment']

# Predictions
y_pred = model.predict(X)

# Confusion Matrix
cm = confusion_matrix(y, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.savefig("outputs/confusion_matrix.png")
plt.show()