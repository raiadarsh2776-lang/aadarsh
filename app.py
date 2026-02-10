import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# APP TITLE
# -----------------------------
st.set_page_config(page_title="ML Practicals", layout="wide")
st.title("Machine Learning Practicals Deployment")

# =============================
# PRACTICAL 4: IMDB SENTIMENT ANALYSIS
# =============================
st.header("📊 Practical 4: IMDB Sentiment Analysis")

uploaded_file = st.file_uploader("Upload IMDB Dataset CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Expecting columns: review, sentiment
    df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})

    st.subheader("Dataset Preview")
    st.write(df.head())

    X_train, X_test, y_train, y_test = train_test_split(
        df["review"],
        df["sentiment"],
        test_size=0.2,
        random_state=42
    )

    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    nb = MultinomialNB()
    nb.fit(X_train_tfidf, y_train)

    y_pred = nb.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    st.success(f"Model Accuracy: {acc:.4f}")

    # Confusion Matrix
    fig, ax = plt.subplots()
    sns.heatmap(
        confusion_matrix(y_test, y_pred),
        annot=True,
        fmt="d",
        cmap="mako",
        ax=ax
    )
    ax.set_title("Sentiment Analysis Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # User Review Prediction
    st.subheader("📝 Test Your Own Review")
    user_review = st.text_input("Enter a movie review")

    if user_review:
        vec = tfidf.transform([user_review])
        prediction = nb.predict(vec)[0]
        st.write(
            f"Sentiment: **{'Positive 😊' if prediction == 1 else 'Negative 😞'}**"
        )

# =============================
# PRACTICAL 5: DIABETES PROGRESSION PREDICTION
# =============================
st.header("🩺 Practical 5: Diabetes Progression Prediction")

diabetes = load_diabetes()

X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data,
    diabetes.target,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f"📉 Mean Squared Error: **{mse:.2f}**")
st.write(f"📈 R-Squared Score: **{r2:.2f}**")

# Plot
fig2, ax2 = plt.subplots()
ax2.scatter(y_test, y_pred, alpha=0.5)
ax2.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)
ax2.set_xlabel("Actual Values")
ax2.set_ylabel("Predicted Values")
ax2.set_title("Diabetes Prediction: Actual vs Predicted")
st.pyplot(fig2)
