import pandas as pd

# Load raw dataset
df = pd.read_csv("data/raw_data.csv")

print("RAW DATA:")
print(df.head())
print("\nSentiment Count:")
print(df['sentiment'].value_counts())


# Load cleaned dataset
df = pd.read_csv("data/cleaned_data.csv")

print("\nCLEANED DATA:")
print(df[['text', 'cleaned_text']].head(10))