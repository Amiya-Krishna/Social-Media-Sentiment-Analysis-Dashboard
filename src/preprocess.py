import re
import nltk
from nltk.corpus import stopwords

# Download stopwords (only first time)
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    
    # Remove special characters & numbers
    text = re.sub(r"[^A-Za-z\s]", "", text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra spaces
    text = text.strip()
    
    # Remove stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    return " ".join(words)