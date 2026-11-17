import re
import nltk
import joblib
import ollama

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# -----------------------------------
# 1. Download NLTK resources
# -----------------------------------


# -----------------------------------
# 2. Load ML Model
# -----------------------------------

model = joblib.load("../model/sentiment_model.pkl")
vectorizer = joblib.load("../model/tfidf_vectorizer.pkl")


# -----------------------------------
# 3. Stopwords
# -----------------------------------

stop_words = set(stopwords.words("english"))


# -----------------------------------
# 4. Text Cleaning
# -----------------------------------

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    words = word_tokenize(text)

    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# -----------------------------------
# 5. Aspect Keywords
# -----------------------------------

ASPECT_KEYWORDS = {

    "Product Quality": [
        "quality",
        "product",
        "material",
        "design",
        "build",
        "durable",
        "broken",
        "damaged"
    ],

    "Delivery": [
        "delivery",
        "delivered",
        "shipping",
        "shipment",
        "late",
        "delay",
        "arrived"
    ],

    "Customer Service": [
        "support",
        "customer service",
        "representative",
        "help",
        "response",
        "complaint"
    ],

    "Price": [
        "price",
        "cost",
        "expensive",
        "cheap",
        "value",
        "money"
    ],

    "Packaging": [
        "package",
        "packaging",
        "box",
        "packed"
    ],

    "Performance": [
        "performance",
        "working",
        "works",
        "stopped",
        "battery",
        "speed",
        "slow"
    ]
}


# -----------------------------------
# 6. Find Aspects
# -----------------------------------

def find_aspects(feedback):

    feedback_lower = feedback.lower()

    detected_aspects = []

    for aspect, keywords in ASPECT_KEYWORDS.items():

        for keyword in keywords:

            if keyword in feedback_lower:
                detected_aspects.append(aspect)
                break

    return detected_aspects


# -----------------------------------
# 7. Generate AI Insights
# -----------------------------------
def generate_ai_insights(feedback, sentiment, aspects):

    if aspects:
        aspect_text = ", ".join(aspects)
    else:
        aspect_text = "No specific aspect detected"

    prompt = f"""
You are an AI customer feedback analysis assistant.

Analyze the following customer feedback.

Customer Feedback:
{feedback}

Overall Machine Learning Sentiment:
{sentiment}

Detected Aspects:
{aspect_text}

For EACH detected aspect, determine whether the customer expresses:
- Positive
- Negative
- Neutral

Then provide:

1. Overall Summary
2. Aspect-Level Sentiment
3. Positive Points
4. Key Issues
5. Suggested Business Action
6. Suggested Response to the Customer

Use this format:

Overall Summary:
<short summary>

Aspect-Level Sentiment:
- <Aspect>: <Positive/Negative/Neutral> - <short reason>
- <Aspect>: <Positive/Negative/Neutral> - <short reason>

Positive Points:
- <point>

Key Issues:
- <issue>

Suggested Business Action:
- <action>

Suggested Response to the Customer:
<short professional response>

Important:
- Base the analysis only on the customer feedback.
- Do not invent facts.
- If an aspect is not clearly discussed, say Neutral.
- Keep the response concise.
"""

    response = ollama.chat(
        model="llama3.1",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


# -----------------------------------
# 8. Get Customer Feedback
# -----------------------------------

feedback = input(
    "\nEnter customer feedback: "
)


# -----------------------------------
# 9. NLP Processing
# -----------------------------------

cleaned_feedback = clean_text(feedback)


# -----------------------------------
# 10. Sentiment Prediction
# -----------------------------------

feedback_vector = vectorizer.transform(
    [cleaned_feedback]
)

sentiment = model.predict(
    feedback_vector
)[0]


# -----------------------------------
# 11. Prediction Confidence
# -----------------------------------

probabilities = model.predict_proba(
    feedback_vector
)[0]

classes = model.classes_


# -----------------------------------
# 12. Aspect Detection
# -----------------------------------

aspects = find_aspects(feedback)


# -----------------------------------
# 13. Generate AI Insights
# -----------------------------------

print("\nGenerating AI insights using Llama 3.1...")

ai_insights = generate_ai_insights(
    feedback,
    sentiment,
    aspects
)


# -----------------------------------
# 14. Display Complete Analysis
# -----------------------------------

print("\n")
print("==============================================")
print("       AI CUSTOMER FEEDBACK ANALYZER")
print("==============================================")


print("\nCustomer Feedback:")
print(feedback)


print("\nSentiment:")
print(sentiment.upper())


print("\nPrediction Confidence:")

for class_name, probability in zip(
    classes,
    probabilities
):

    print(
        f"{class_name.capitalize():10} : "
        f"{probability * 100:.2f}%"
    )


print("\nDetected Aspects:")

if aspects:

    for aspect in aspects:
        print("-", aspect)

else:

    print("- No specific aspect detected")


print("\n----------------------------------------------")
print("GENERATIVE AI INSIGHTS")
print("----------------------------------------------")

print(ai_insights)


print("\n==============================================")