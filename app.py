import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
st.title("Titanic Survival Prediction")
file = st.file_uploader("Upload Titanic CSV", type="csv")

if file:
    df = pd.read_csv(file)
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df = df.dropna(subset=["Embarked"])
    df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)
    features = [
        "Pclass", "Age", "SibSp", "Parch", "Fare", 
        "Sex_male", "Embarked_Q", "Embarked_S"
    ]
    if all(col in df.columns for col in features):
        X = df[features]
        y = df["Survived"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        st.subheader("Model Performance")
        acc = accuracy_score(y_test, y_pred)
        st.write(f"**Accuracy:** {acc:.2f}")
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap="Blues", ax=ax)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)
    else:
        st.error("The uploaded CSV is missing required columns (Sex, Embarked, etc.)")
