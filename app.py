import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="IMDB Sentiment Analysis", layout="centered")
st.title("🎬 IMDB Sentiment Analysis")

uploaded_file = st.file_uploader("Upload IMDB Dataset CSV", type="csv")

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)

    # Clean column names
    df.columns = df.columns.str.strip().str.lower()

    # Detect sentiment column
    sentiment_col = None
    for col in ["sentiment", "label", "polarity"]:
        if col in df.columns:
            sentiment_col = col
            break

    if sentiment_col is None:
        st.error("❌ No sentiment column found. Expected: sentiment / label / polarity")
        st.stop()

    # Detect review/text column
    text_col = None
    for col in ["review", "text", "sentence", "comment"]:
        if col in df.columns:
            text_col = col
            break

    if text_col is None:
        st.error("❌ No text column found. Expected: review / text / sentence / comment")
        st.stop()

    # Map sentiment values
    df[sentiment_col] = df[sentiment_col].astype(str).str.lower()
    df = df[df[sentiment_col].isin(["positive", "negative"])]

    df[sentiment_col] = df[sentiment_col].map({
        "positive": 1,
        "negative": 0
    })

    st.subheader("📄 Dataset Preview")
    st.dataframe(df[[text_col, sentiment_col]].head())

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col],
        df[sentiment_col],
        test_size=0.2,
        random_state=42,
        stratify=df[sentiment_col]
    )

    # TF-IDF Vectorizer
    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Train Naive Bayes model
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    # Predictions
    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    st.success(f"✅ Model Accuracy: {accuracy:.4f}")

    # Confusion Matrix
    fig, ax = plt.subplots()
    sns.heatmap(
        confusion_matrix(y_test, y_pred),
        annot=True,
        fmt="d",
        cmap="mako",
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

    # User input prediction
    st.subheader("🧪 Test Your Own Review")
    user_review = st.text_area("Enter a movie review:")

    if user_review.strip():
        vec = tfidf.transform([user_review])
        prediction = model.predict(vec)[0]
        probability = model.predict_proba(vec).max()

        if prediction == 1:
            st.success(f"😊 Positive Review (Confidence: {probability:.2%})")
        else:
            st.error(f"😞 Negative Review (Confidence: {probability:.2%})")

else:
    st.info("👆 Upload an IMDB CSV file to begin")
