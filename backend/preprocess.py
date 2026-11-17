import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

# Load dataset
data = pd.read_csv("../dataset/customer_feedback.csv")

# Get English stopwords
stop_words = set(stopwords.words("english"))


# Function to clean feedback
def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation and special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Tokenize
    words = word_tokenize(text)

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    # Join words back together
    cleaned_text = " ".join(words)

    return cleaned_text


# Apply preprocessing
data["cleaned_feedback"] = data["feedback"].apply(clean_text)

# Display original and cleaned feedback
print("\nOriginal vs Cleaned Feedback:\n")

for i in range(len(data)):
    print("Original :", data.loc[i, "feedback"])
    print("Cleaned  :", data.loc[i, "cleaned_feedback"])
    print("-" * 70)