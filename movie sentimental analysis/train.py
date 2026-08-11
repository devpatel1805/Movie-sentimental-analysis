import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
from preprocess import clean_text

# Load CSV
df = pd.read_csv("/Users/pateldev/Downloads/movie sentimental analysis/IMDB Dataset 2.csv")

# Use correct column names
df["clean_text"] = df["review"].apply(clean_text)

# Convert text → numbers
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["clean_text"])

# Map string sentiment to numbers
df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})

# Labels
y = df["sentiment"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

import os
os.makedirs(r"/Users/pateldev/Downloads/movie sentimental analysis/model", exist_ok=True)

# Save model
joblib.dump(model, r"/Users/pateldev/Downloads/movie sentimental analysis/model/saved_model.pkl")
joblib.dump(vectorizer, r"/Users/pateldev/Downloads/movie sentimental analysis/model/tfidf_vectorizer.pkl")


print("Training complete!")
print("Accuracy:", model.score(X_test, y_test))
