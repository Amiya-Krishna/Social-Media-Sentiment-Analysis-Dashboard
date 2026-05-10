# 📊 Social Media Sentiment Analysis Dashboard

## 🚀 Project Overview
This project is a Machine Learning + NLP-based web application that analyzes social media text (tweets, comments, reviews) and classifies sentiment into:

- Positive 😊  
- Negative 😡  
- Neutral 😐  

It also includes an interactive dashboard built using Streamlit.

---

## 🎯 Problem Statement
Businesses receive thousands of customer comments daily. Manual analysis is slow and inefficient. This system automates sentiment detection to help companies understand customer emotions in real time.

---

## 🧠 Solution
We built an end-to-end pipeline:
- Data preprocessing
- Text cleaning
- TF-IDF feature extraction
- Logistic Regression model
- Streamlit dashboard

---

## 🛠 Tech Stack
- Python
- Pandas, NumPy
- NLTK
- Scikit-learn
- TF-IDF
- Logistic Regression
- Streamlit
- Plotly

---

## 🏗 Project Architecture
Data → Cleaning → Preprocessing → TF-IDF → ML Model → Prediction → Dashboard


---

## 📂 Folder Structure
data/ → Dataset files
src/ → ML pipeline code
models/ → Trained model files
app/ → Streamlit dashboard
images/ → Screenshots
outputs/ → Results & graphs


---

## ⚙️ Installation

```bash
git clone <repo-link>
cd Social-Media-Sentiment-Analysis-Dashboard
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 🚀 How to Run
Train Model
```bash
python src/train.py
```
Run Dashboard
```bash
streamlit run app/app.py
```

## 📊 Features
Real-time sentiment prediction
CSV upload analysis
Interactive charts
Clean UI dashboard
Downloadable results

## 📸 Screenshots
Dashboard

(Add image here)

Sentiment Distribution

(Add image here)

## 📈 Results
Accuracy: ~85–95%
Balanced dataset performance
Fast prediction pipeline

## 💡 Learning Outcomes
NLP preprocessing
TF-IDF vectorization
ML classification
Model deployment using Streamlit
End-to-end ML pipeline design

## 👨‍💻 Author

Amiya Krishna Chaurasiya
Engineering Student
Machine Learning & NLP Enthusiast