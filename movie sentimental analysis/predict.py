import joblib
from preprocess import clean_text  # 1. Import your cleaning function

# Load model and vectorizer separately
model = joblib.load(r"/Users/pateldev/Downloads/movie sentimental analysis/model/saved_model.pkl")
vectorizer = joblib.load(r"/Users/pateldev/Downloads/movie sentimental analysis/model/tfidf_vectorizer.pkl")

text = input("Enter a review: ")

# 2. Clean the input text first
cleaned = clean_text(text)

# 3. Convert input text to TF-IDF vector
vectorized_text = vectorizer.transform([cleaned])

# 4. Predict sentiment
prediction = model.predict(vectorized_text)[0]

# 5. Compare with string labels (or lowercased string)
if prediction == 1:
    print("Sentiment: Positive 😀")
else:
    print("Sentiment: Negative 🙁")