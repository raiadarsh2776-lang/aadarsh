import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("Spam Email Detector")

emails = [
    "Win a free iPhone now",
    "Meeting at 11 am tomorrow",
    "Congratulations you won lottery",
    "Project discussion with team",
    "Claim your prize immediately",
    "Please find the attached report",
    "Limited offer buy now",
    "Urgent offer expires today",
    "Schedule the meeting for Monday",
    "You have won a cash prize",
    "Monthly performance report attached",
    "Exclusive deal just for you"
]

labels = [1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1]

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_df=0.9,
    min_df=1
)

X = vectorizer.fit_transform(emails)

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.25, random_state=42, stratify=labels
)

model = LinearSVC(C=1.0, random_state=42)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
st.write(f"Model Accuracy: {accuracy:.2f}")

# --- User Input Section ---
user_msg = st.text_area("Enter Email Message")

if st.button("Check"):
    msg_vec = vectorizer.transform([user_msg])
    pred = model.predict(msg_vec)[0]

    if pred == 1:
        st.write("Result: Spam Email 🚫")
    else:
        st.write("Result: Not Spam Email ✅")
















# import streamlit as st
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import accuracy_score, confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt

# st.title("IMDB Sentiment Analysis")

# uploaded_file = st.file_uploader("Upload IMDB Dataset CSV", type="csv")

# if uploaded_file:
#     df = pd.read_csv(uploaded_file)
#     df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})

#     st.write(df.head())

#     X_train, X_test, y_train, y_test = train_test_split(
#         df["review"], df["sentiment"], test_size=0.2, random_state=42
#     )

#     tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
#     X_train_tfidf = tfidf.fit_transform(X_train)
#     X_test_tfidf = tfidf.transform(X_test)

#     nb = MultinomialNB()
#     nb.fit(X_train_tfidf, y_train)

#     y_pred = nb.predict(X_test_tfidf)

#     st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

#     fig, ax = plt.subplots()
#     sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
#     ax.set_title("Sentiment Analysis Confusion Matrix")
#     st.pyplot(fig)

#     user_review = st.text_input("Enter a Review")

#     if user_review:
#         vec = tfidf.transform([user_review])
#         result = nb.predict(vec)[0]
#         st.write("Sentiment:", "Positive" if result == 1 else "Negative")




# # import streamlit as st
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from sklearn.datasets import load_diabetes
# # from sklearn.model_selection import train_test_split
# # from sklearn.linear_model import LinearRegression
# # from sklearn.metrics import mean_squared_error, r2_score

# # st.title("Diabetes Progression Regression")

# # diabetes = load_diabetes()

# # X_train, X_test, y_train, y_test = train_test_split(
# #     diabetes.data,
# #     diabetes.target,
# #     test_size=0.2,
# #     random_state=42
# # )

# # model = LinearRegression()
# # model.fit(X_train, y_train)

# # y_pred = model.predict(X_test)

# # st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
# # st.write(f"R-squared: {r2_score(y_test, y_pred):.2f}")

# # fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# # axs[0].scatter(y_test, y_pred, alpha=0.5)
# # axs[0].plot(
# #     [y_test.min(), y_test.max()],
# #     [y_test.min(), y_test.max()],
# #     linestyle="--"
# # )
# # axs[0].set_title("True vs Predicted Values")
# # axs[0].set_xlabel("True Values")
# # axs[0].set_ylabel("Predicted Values")

# # # Note: X_test[:, 2] typically represents the BMI feature in this dataset
# # axs[1].scatter(X_test[:, 2], y_pred, alpha=0.7)
# # axs[1].set_title("Feature (BMI) vs Predicted Values")
# # axs[1].set_xlabel("BMI Feature")
# # axs[1].set_ylabel("Predicted Progression")

# # st.pyplot(fig)

