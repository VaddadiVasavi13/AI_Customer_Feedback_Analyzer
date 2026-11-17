import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# 1. Load Dataset
# ==========================================

print("Loading dataset...")

df = pd.read_csv("dataset/Dataset-SA.csv")

print("Total records:", len(df))


# ==========================================
# 2. Keep Required Columns
# ==========================================

df = df[["Review", "Summary", "Sentiment"]].copy()


# ==========================================
# 3. Handle Missing Text
# ==========================================

df["Review"] = df["Review"].fillna("")
df["Summary"] = df["Summary"].fillna("")


# ==========================================
# 4. Combine Review + Summary
# ==========================================

df["text"] = (
    df["Review"].astype(str)
    + " "
    + df["Summary"].astype(str)
)


# ==========================================
# 5. Clean Text
# ==========================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


print("Cleaning text...")

df["text"] = df["text"].apply(clean_text)


# ==========================================
# 6. Remove Empty Text
# ==========================================

df = df[df["text"].str.len() > 0]


# ==========================================
# 7. Balance Dataset
# ==========================================

print("\nOriginal distribution:")
print(df["Sentiment"].value_counts())


positive = df[df["Sentiment"] == "positive"].sample(
    n=28000,
    random_state=42
)

negative = df[df["Sentiment"] == "negative"].sample(
    n=28000,
    random_state=42
)

neutral = df[df["Sentiment"] == "neutral"].sample(
    n=10000,
    random_state=42
)


balanced_df = pd.concat(
    [positive, negative, neutral],
    ignore_index=True
)


# Shuffle
balanced_df = balanced_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


print("\nBalanced distribution:")
print(balanced_df["Sentiment"].value_counts())


# ==========================================
# 8. Input and Output
# ==========================================

X = balanced_df["text"]
y = balanced_df["Sentiment"]


# ==========================================
# 9. Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 10. TF-IDF
# ==========================================

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)


print(
    "TF-IDF training shape:",
    X_train_tfidf.shape
)


# ==========================================
# 11. Train Logistic Regression
# ==========================================

print("\nTraining Logistic Regression...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(
    X_train_tfidf,
    y_train
)


# ==========================================
# 12. Evaluate
# ==========================================

print("\nEvaluating model...")

predictions = model.predict(X_test_tfidf)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n========================================")
print("MODEL RESULTS")
print("========================================")

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions
    )
)


# ==========================================
# 13. Test Negative Feedback
# ==========================================

test_feedback = [
    "The product is terrible. The quality is very poor and the delivery was extremely late.",
    "I am very disappointed with this product. It is useless and customer service was not helpful.",
    "The product is excellent and I am very happy with my purchase.",
    "The product is okay. Nothing special."
]


print("\n========================================")
print("TEST PREDICTIONS")
print("========================================")

test_cleaned = [
    clean_text(text)
    for text in test_feedback
]

test_vectors = vectorizer.transform(
    test_cleaned
)

test_predictions = model.predict(
    test_vectors
)

test_probabilities = model.predict_proba(
    test_vectors
)


for text, prediction, probabilities in zip(
    test_feedback,
    test_predictions,
    test_probabilities
):

    print("\nFeedback:")
    print(text)

    print("Prediction:", prediction)

    print("Confidence:")

    for label, probability in zip(
        model.classes_,
        probabilities
    ):
        print(
            f"  {label}: {probability * 100:.2f}%"
        )


# ==========================================
# 14. Save New Model
# ==========================================

print("\nSaving model...")

joblib.dump(
    model,
    "model/sentiment_model_new.pkl"
)

joblib.dump(
    vectorizer,
    "model/tfidf_vectorizer_new.pkl"
)

print("\n========================================")
print("DONE")
print("========================================")

print("New model saved successfully.")