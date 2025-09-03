# NLP_SentimentAnalysis

# AI Echo: ChatGPT Reviews Sentiment Analysis

## 📖 Overview
AI Echo is a sentiment analysis project designed to analyze user reviews of the ChatGPT application.  
The system uses **Natural Language Processing (NLP)**, **Machine Learning**, and **Deep Learning** to classify reviews as **Positive, Neutral, or Negative**.  

Deployed as a **Streamlit web app**, it provides insights into customer experiences, user feedback patterns, and model predictions.

---

## 🚀 Features
- **Dataset Overview**: Explore review dataset, shape, columns, and distributions.
- **Insights (10 Questions)**: Interactive Q&A on key business questions:
  - Sentiment distribution
  - Trends over time
  - Keywords in positive/negative reviews
  - Verified vs non-verified users
  - Regional, platform, and version-based analysis
- **Prediction Tool**: Enter any review and predict sentiment using a trained model.
- **Model Evaluation**: Displays evaluation metrics and explains limitations (low accuracy due to small dataset).

---

## 🛠️ Tech Stack
- **Language**: Python  
- **Libraries**: Pandas, Numpy, Matplotlib, Seaborn, WordCloud, Scikit-learn  
- **Deployment**: Streamlit  
- **Dataset**: 51 user reviews with ratings, platform, version, etc.  

---

## 📂 Project Structure
📁 Project Root
│── Sentiment_analysis_app.py # Streamlit app
│── Sentiment_Analysis.ipynb # Data preprocessing & model building
│── sentiment_model.pkl # Trained Random Forest model
│── chatgpt_style_reviews_dataset.xlsx # Dataset
│── README.md # Project summary

---

## 📊 Model Limitations
⚠️ The dataset contains only **51 rows**, which limits model accuracy and generalization.  
This is why metrics like accuracy, precision, and recall remain low. More data is needed for robust performance.

---
📌 Author
Rajkumar
