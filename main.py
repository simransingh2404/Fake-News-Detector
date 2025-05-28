import joblib
from fetch_news import fetch_headlines

# Load the trained model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Fetch the latest news headlines (title + description)
headlines = fetch_headlines()

# Predict and display each headline
for headline in headlines:
    vector = vectorizer.transform([headline])
    prediction = model.predict(vector)[0]

    # Determine label
    label = "FAKE" if prediction == 1 else "REAL"

    print(f"\n📰 Headline: {headline}")
    print(f"🔎 Prediction: {label}")
    print("-" * 80)
