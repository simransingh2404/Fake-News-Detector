import pickle
from news_classifier import clean_text  # Ensure this function is defined

# Load the trained model
with open("fake_news_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load the TF-IDF vectorizer
with open("tfidf_vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

# Sample real-time news data (you can modify this or fetch live data)
data = {
    "articles": [
        {"title": "Government announces new education policy for rural areas"},
        {"title": "NASA confirms aliens landed in Rajasthan"},
        {"title": "Scientists discover new water source on Mars"},
        {"title": "Actor replaces PM in unexpected move"}
    ]
}

# Analyze articles
for article in data["articles"]:
    title = article.get("title", "")
    if not title:
        continue

    cleaned_title = clean_text(title)
    vectorized_title = vectorizer.transform([cleaned_title])
    prediction = model.predict(vectorized_title)[0]

    result = "FAKE" if prediction == 0 else "REAL"
    print(f"📰 Title: {title}\n   ➤ Prediction: {result}\n")
