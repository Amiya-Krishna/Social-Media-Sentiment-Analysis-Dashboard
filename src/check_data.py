import pandas as pd

df = pd.read_csv("data/raw_data.csv")

print(df.head())
print(df['sentiment'].value_counts())