import pandas as pd
import random

positive = [
    "I love this product", "Amazing experience", "Great service",
    "Absolutely fantastic", "Very happy with this", "Superb quality"
]

negative = [
    "Worst experience ever", "Very bad service", "Hate this app",
    "Terrible performance", "Not satisfied", "Disappointing"
]

neutral = [
    "It is okay", "Average experience", "Nothing special",
    "It works fine", "Not bad", "Can be improved"
]

data = []

for _ in range(200):
    data.append([random.choice(positive), "positive"])
    data.append([random.choice(negative), "negative"])
    data.append([random.choice(neutral), "neutral"])

df = pd.DataFrame(data, columns=["text", "sentiment"])
df.to_csv("data/raw_data.csv", index=False)

print("Dataset generated successfully!")