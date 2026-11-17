import pandas as pd
import re
import nltk
import joblib

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# -----------------------------------
# 1. Download NLTK resources
# -----------------------------------

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")


# -----------------------------------
# 2. Load Real Dataset
# -----------------------------------

data = pd.read_csv("../dataset/customer_feedback_real.csv")

print("Total reviews:", len(data))


# -----------------------------------
# 3. Text Preprocessing
# -----------------------------------

stop_words = set(stopwords.words("english"))


def clean_text(text):

    text = str(text).lower()

    # Remove punctuation and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Tokenize
    words = word_tokenize(text)

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    return " ".join(words)


print("\nCleaning reviews...")

data["cleaned_feedback"] = data["feedback"].apply(clean_text)


# -----------------------------------
# 4. Separate Features and Labels
# -----------------------------------

X_text = data["cleaned_feedback"]
y = data["sentiment"]


# -----------------------------------
# 5. Split Dataset
# -----------------------------------

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining reviews:", len(X_train_text))
print("Testing reviews:", len(X_test_text))


# -----------------------------------
# 6. TF-IDF
# -----------------------------------

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

# Fit ONLY on training data
X_train = vectorizer.fit_transform(X_train_text)

# Transform test data
X_test = vectorizer.transform(X_test_text)


print("\nNumber of TF-IDF features:", X_train.shape[1])


# -----------------------------------
# 7. Train Logistic Regression
# -----------------------------------

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train, y_train)


# -----------------------------------
# 8. Prediction
# -----------------------------------

y_pred = model.predict(X_test)


# -----------------------------------
# 9. Evaluation
# -----------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n===================================")
print("MODEL RESULTS")
print("===================================")

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# -----------------------------------
# 10. Save Model
# -----------------------------------

joblib.dump(
    model,
    "../model/sentiment_model.pkl"
)

joblib.dump(
    vectorizer,
    "../model/tfidf_vectorizer.pkl"
)


print("\n===================================")
print("Model saved successfully!")
print("===================================")