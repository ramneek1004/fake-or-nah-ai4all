import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -------------------------------------------------
# Train a tiny demo model (based on fake/real news)
# -------------------------------------------------
@st.cache_resource
def train_model():
    # Very small toy dataset just for demonstration.
    # This keeps things fast for Streamlit deployment.
    texts = [
        "Breaking: President signs major healthcare bill into law",
        "Local school district announces new lunch program for students",
        "Scientists discover water beneath the surface of Mars",
        "Central bank raises interest rates to combat inflation",
        "You won’t believe what this celebrity said about vaccines",
        "Click here now to claim your free iPhone",
        "Government secretly replacing money with digital chips",
        "Doctors hate her: one simple trick to cure cancer"
    ]

    # 1 = real-ish, 0 = fake-ish
    labels = [1, 1, 1, 1, 0, 0, 0, 0]

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(texts)

    model = LogisticRegression()
    model.fit(X, labels)

    return model, vectorizer


model, vectorizer = train_model()

# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------
st.title("Fake or Nah? – Fake News Detector")

st.write("""
This web app is part of our **Fake or Nah?** project for detecting fake news.  
It uses a small machine learning model trained on example fake and real headlines
to predict whether new text looks **Real** or **Fake**.
""")

user_text = st.text_area(
    "Paste a news headline or short article below:",
    placeholder="Example: 'Scientists discover evidence of life on Mars'"
)

if st.button("Predict"):
    if not user_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        X_new = vectorizer.transform([user_text])
        pred = model.predict(X_new)[0]
        proba = model.predict_proba(X_new)[0]

        label = "🟢 Real-looking news" if pred == 1 else "🔴 Fake-looking news"

        st.subheader("Prediction")
        st.write(f"**{label}**")

        st.subheader("Model confidence")
        st.write(f"Real: {proba[1]:.2f}   |   Fake: {proba[0]:.2f}")

st.caption("Note: This is a simple demo model, not a production-grade fact checker.")
