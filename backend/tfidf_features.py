import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK data
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

# Load dataset
data = pd.read_csv("../dataset/customer_feedback.csv")

# Get English stopwords
stop_words = set(stopwords.words("english"))


# Text cleaning function
def clean_text(text):

    text = text.lower()

    # Remove punctuation and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    words = word_tokenize(text)

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    return " ".join(words)


# Clean the feedback
data["cleaned_feedback"] = data["feedback"].apply(clean_text)


# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Convert text into numerical features
X = vectorizer.fit_transform(data["cleaned_feedback"])

# Get sentiment labels
y = data["sentiment"]


print("Original number of feedback records:", len(data))

print("Number of numerical features:", X.shape[1])

print("\nFeature matrix shape:")
print(X.shape)

print("\nSome important words/features:")
print(vectorizer.get_feature_names_out())

print("\nSentiment labels:")
print(y.values)