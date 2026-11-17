import pandas as pd

# Load the real Amazon reviews dataset
data = pd.read_csv("../dataset/amazon_reviews.csv")

print("Original dataset size:", len(data))

# Keep only the columns we need
data = data[["reviewText", "overall"]]

# Remove reviews with missing text
data = data.dropna(subset=["reviewText"])

# Convert ratings into sentiment
def get_sentiment(rating):

    if rating <= 2:
        return "negative"

    elif rating == 3:
        return "neutral"

    else:
        return "positive"


data["sentiment"] = data["overall"].apply(get_sentiment)

# Rename review column
data = data.rename(columns={
    "reviewText": "feedback",
    "overall": "rating"
})

# Save the cleaned dataset
data.to_csv(
    "../dataset/customer_feedback_real.csv",
    index=False
)

# Display results
print("\nCleaned dataset size:", len(data))

print("\nSentiment distribution:")
print(data["sentiment"].value_counts())

print("\nFirst 10 records:")
print(data.head(10))

print("\nClean dataset saved successfully!")