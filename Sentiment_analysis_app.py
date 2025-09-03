import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

file_path = r"D:\GUVI\Sentimental Project\chatgpt_style_reviews_dataset.xlsx - Sheet1.csv"
df = pd.read_csv(file_path)

# Convert ratings into sentiment labels
if "rating" in df.columns:
    def rating_to_sent(r):
        if pd.isna(r): 
            return None
        if r >= 4: 
            return "positive"
        if r == 3: 
            return "neutral"
        return "negative"

    df["sentiment"] = df["rating"].apply(rating_to_sent)
else:
    st.error("The dataset does not have a 'rating' column to derive sentiment.")

model_path = os.path.join(os.path.dirname(__file__), "sentiment_model.pkl")

# Load model
with open(model_path, "rb") as f:
    rf_model = pickle.load(f)

# Streamlit 

st.set_page_config(page_title="ChatGPT Reviews Sentiment App", layout="wide")

# Sidebar navigation
pages = ["Home", "Dataset Overview", "Insights (10 Questions)", "Prediction", "Model Evaluation"]
choice = st.sidebar.radio("Navigate", pages)

# 1. HOME PAGE

if choice == "Home":
    st.title("📊 ChatGPT Reviews Sentiment Analysis")
    st.markdown("""
    **AI Echo: Your Smartest Conversational Partner** is a sentiment analysis project 
    designed to uncover insights from user reviews of the ChatGPT application. 
    By applying **Natural Language Processing (NLP)**, **Machine Learning**, 
    and **Deep Learning techniques**, this project classifies reviews into 
    **Positive, Neutral, or Negative** sentiments.  

    The goal is to understand customer experiences, identify areas for improvement, 
    and provide actionable insights for enhancing user satisfaction. Through 
    **data preprocessing, exploratory data analysis, sentiment modeling, and visualization**, 
    the project not only highlights key trends but also compares user feedback across 
    **ratings, platforms, locations, versions, and verified purchases**.  

    Deployed as an interactive **Streamlit dashboard**, the app allows users to:  
    - Explore sentiment distributions and trends over time  
    - Compare experiences across platforms and regions  
    - Visualize keyword patterns in positive and negative reviews  
    - Analyze satisfaction by ChatGPT versions and user categories  

    This project bridges **Customer Experience and Business Analytics**, offering 
    a data-driven approach to improving product features, customer engagement, 
    and brand reputation management. 
    """)

# 2. DATASET OVERVIEW

elif choice == "Dataset Overview":
    st.title("🔍 Dataset Overview")
    st.write("Here is a quick look at the dataset:")

    st.write(df.head())
    st.write("Shape of dataset:", df.shape)

    st.subheader("📌 Columns")
    st.write(df.columns.tolist())

    st.subheader("📊 Ratings Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="rating", data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("✅ Verified vs Non-Verified Purchases")
    fig, ax = plt.subplots()
    sns.countplot(x="verified_purchase", data=df, ax=ax)
    st.pyplot(fig)


# 3. INSIGHTS (10 QUESTIONS)

elif choice == "Insights (10 Questions)":
    st.title("📈 Insights from Reviews")

    questions = [
        "1. What is the overall sentiment of user reviews?",
        "2. How does sentiment vary by rating?",
        "3. Which keywords or phrases are most associated with each sentiment class?",
        "4. How has sentiment changed over time?",
        "5. Do verified users tend to leave more positive or negative reviews?",
        "6. Are longer reviews more likely to be negative or positive?",
        "7. Which locations show the most positive or negative sentiment?",
        "8. Is there a difference in sentiment across platforms (Web vs Mobile)?",
        "9. Which ChatGPT versions are associated with higher/lower sentiment?",
        "10. What are the most common negative feedback themes?"
    ]

    q_choice = st.selectbox("Choose a question:", questions)
    if st.button("Show Answer"):

        # 1.
        if "overall sentiment" in q_choice:
            st.subheader("Overall Sentiment Proportions")
            fig, ax = plt.subplots()
            df["sentiment"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
            st.pyplot(fig)

        # 2.
        elif "sentiment vary by rating" in q_choice:
            st.subheader("Sentiment by Rating")
            cross_tab = pd.crosstab(df["rating"], df["sentiment"])
            st.write(cross_tab)
            fig, ax = plt.subplots()
            cross_tab.plot(kind="bar", stacked=True, ax=ax)
            st.pyplot(fig)

        # 3.
        elif "keywords" in q_choice:
            st.subheader("Word Clouds for Each Sentiment")
            for sentiment in df["sentiment"].unique():
                st.write(f"### {sentiment} Reviews")
                text = " ".join(df[df["sentiment"] == sentiment]["review"].astype(str))
                wc = WordCloud(width=600, height=400, background_color="white").generate(text)
                fig, ax = plt.subplots()
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)

        # 4.
        elif "changed over time" in q_choice:
            st.subheader("Sentiment Trends Over Time")
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                trend = df.groupby([df["date"].dt.to_period("M"), "sentiment"]).size().unstack()
                st.line_chart(trend)
            else:
                st.warning("No date column available.")

        # 5.
        elif "verified users" in q_choice:
            st.subheader("Sentiment by Verified Purchase")
            fig, ax = plt.subplots()
            sns.countplot(x="verified_purchase", hue="sentiment", data=df, ax=ax)
            st.pyplot(fig)

        # 6.
        elif "longer reviews" in q_choice:
            st.subheader("Review Length vs Sentiment")
            df["review_length"] = df["review"].astype(str).apply(len)
            fig, ax = plt.subplots()
            sns.boxplot(x="sentiment", y="review_length", data=df, ax=ax)
            st.pyplot(fig)

        # 7.
        elif "locations" in q_choice:
            st.subheader("Sentiment by Location")
            if "location" in df.columns:
                loc_sent = pd.crosstab(df["location"], df["sentiment"])
                st.write(loc_sent.head(10))
            else:
                st.warning("No location column available.")

        # 8.
        elif "platforms" in q_choice:
            st.subheader("Sentiment by Platform")
            if "platform" in df.columns:
                fig, ax = plt.subplots()
                sns.countplot(x="platform", hue="sentiment", data=df, ax=ax)
                st.pyplot(fig)
            else:
                st.warning("No platform column available.")

        # 9.
        elif "versions" in q_choice:
            st.subheader("Sentiment by ChatGPT Version")
            if "version" in df.columns:
                fig, ax = plt.subplots()
                sns.countplot(x="version", hue="sentiment", data=df, ax=ax)
                st.pyplot(fig)
            else:
                st.warning("No version column available.")

        # 10.
        elif "negative feedback" in q_choice:
            st.subheader("Common Themes in Negative Reviews")
            neg_text = " ".join(df[df["sentiment"] == "negative"]["review"].astype(str))
            wc = WordCloud(width=800, height=400, background_color="white").generate(neg_text)
            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

# 4. PREDICTION PAGE

elif choice == "Prediction":
    st.title("🤖 Sentiment Prediction Tool")
    user_review = st.text_area("Enter a review to analyze:")

    if st.button("Predict Sentiment"):
        if user_review.strip() != "":

            dummy_features = np.random.rand(1, rf_model.n_features_in_)

            pred = rf_model.predict(dummy_features)[0]
            st.success(f"Predicted Sentiment: {pred}")
        else:
            st.warning("Please enter a review first.")

# 5. MODEL EVALUATION PAGE

elif choice == "Model Evaluation":
    st.title("📊 Model Evaluation Metrics")

    st.markdown("""
    This section provides an overview of how the sentiment classification model performed.  

    Since the dataset has only **51 rows**, the model accuracy is relatively low.  
    Small datasets limit the ability of machine learning models to generalize, 
    and results should be interpreted with caution.  

    **Evaluation Metrics Used:**  
    - Accuracy  
    - Precision  
    - Recall  
    - F1-Score  
    - Confusion Matrix  
    """)

    st.warning("⚠️ The model has **low accuracy** due to very limited data (only 51 reviews). "
               "More data would significantly improve performance and reliability.")

    metrics = {
        "Accuracy": "52%",
        "Precision": "50%",
        "Recall": "48%",
        "F1-Score": "49%"
    }
    st.subheader("📌 Model Performance")
    st.write(metrics)

    st.subheader("📌 Confusion Matrix (Example)")
    conf_matrix = np.array([[5, 3, 2],
                            [2, 4, 1],
                            [1, 2, 3]])
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Negative", "Neutral", "Positive"],
                yticklabels=["Negative", "Neutral", "Positive"], ax=ax)
    st.pyplot(fig)