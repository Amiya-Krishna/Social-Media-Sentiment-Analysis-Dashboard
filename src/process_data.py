import pandas as pd
from preprocess import clean_text

# Load dataset
df = pd.read_csv("data/raw_data.csv")

# Apply cleaning
df['cleaned_text'] = df['text'].apply(clean_text)

# Save cleaned dataset
df.to_csv("data/cleaned_data.csv", index=False)

print("Data cleaned and saved successfully!")