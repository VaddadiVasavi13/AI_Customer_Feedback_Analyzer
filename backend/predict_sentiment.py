import re
import nltk
import joblib

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# Download NLTK resources
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")


# Load trained model and TF-IDF vectorizer
model = joblib.load("../model/sentiment_model.pkl")
vectorizer = joblib.load("../model/tfidf_vectorizer.pkl")


# Stopwords
stop_words = set(stopwords.words("english"))


# -----------------------------------
# Text Cleaning
# -----------------------------------

def clean_text(text):

    text = str(text).lower()

    # Remove punctuation and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Tokenize
    words = word_tokenize(text)

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    return " ".join(words)


# -----------------------------------
# Get User Feedback
# -----------------------------------

feedback = input("\nEnter customer feedback: ")


# Clean the feedback
cleaned_feedback = clean_text(feedback)


# Convert to TF-IDF
feedback_vector = vectorizer.transform([cleaned_feedback])


# Predict sentiment
prediction = model.predict(feedback_vector)[0]


# Get prediction probabilities
probabilities = model.predict_proba(feedback_vector)[0]

classes = model.classes_


print("\n===================================")
print("CUSTOMER FEEDBACK ANALYSIS")
print("===================================")

print("\nOriginal Feedback:")
print(feedback)

print("\nCleaned Feedback:")
print(cleaned_feedback)

print("\nPredicted Sentiment:")
print(prediction.upper())

print("\nPrediction Confidence:")

for class_name, probability in zip(classes, probabilities):
    print(f"{class_name.capitalize():10} : {probability * 100:.2f}%")

print("\n===================================")