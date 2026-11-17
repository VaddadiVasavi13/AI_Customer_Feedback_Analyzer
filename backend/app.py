from flask import Flask, request, jsonify
from flask_cors import CORS

import re
import joblib
import ollama

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pathlib import Path

# ==========================================
# Create Flask Application
# ==========================================

app = Flask(__name__)

CORS(app)


# ==========================================
# Load ML Model and TF-IDF Vectorizer
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR.parent / "model"

model = joblib.load(
    MODEL_DIR / "sentiment_model.pkl"
)

vectorizer = joblib.load(
    MODEL_DIR / "tfidf_vectorizer.pkl"
)

# ==========================================
# Stopwords
# ==========================================

stop_words = set(stopwords.words("english"))


# ==========================================
# Text Cleaning
# ==========================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    words = word_tokenize(text)

    words = [
        word
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# ==========================================
# Aspect Keywords
# ==========================================

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


# ==========================================
# Find Aspects
# ==========================================

def find_aspects(feedback):

    feedback_lower = feedback.lower()

    detected_aspects = []

    for aspect, keywords in ASPECT_KEYWORDS.items():

        for keyword in keywords:

            if keyword in feedback_lower:

                detected_aspects.append(aspect)

                break

    return detected_aspects


# ==========================================
# Generate AI Insights
# ==========================================

def generate_ai_insights(
    feedback,
    sentiment,
    aspects
):

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

@app.route("/", methods=["GET"])
def home():
    return "AI Customer Feedback Analyzer API is running!"


# ==========================================
# API Route
# ==========================================
@app.route("/analyze", methods=["POST"])
def analyze_feedback():

    try:

        # Get JSON data
        data = request.get_json()

        # Get feedback
        feedback = data.get("feedback", "").strip()


        # Check empty input
        if not feedback:

            return jsonify({
                "error": "Please enter customer feedback."
            }), 400


        # ==================================
        # Clean Feedback
        # ==================================

        cleaned_feedback = clean_text(feedback)


        # ==================================
        # Sentiment Prediction
        # ==================================

        feedback_vector = vectorizer.transform(
            [cleaned_feedback]
        )

        sentiment = model.predict(
            feedback_vector
        )[0]


        # ==================================
        # Prediction Confidence
        # ==================================

        probabilities = model.predict_proba(
            feedback_vector
        )[0]

        classes = model.classes_

        confidence = {}

        for class_name, probability in zip(
            classes,
            probabilities
        ):

            confidence[class_name] = round(
                float(probability) * 100,
                2
            )


        # ==================================
        # Aspect Detection
        # ==================================

        aspects = find_aspects(feedback)


        # ==================================
        # Generative AI
        # ==================================

        ai_insights = generate_ai_insights(
            feedback,
            sentiment,
            aspects
        )


        # ==================================
        # Return Result
        # ==================================

        return jsonify({

            "feedback": feedback,

            "sentiment": sentiment,

            "confidence": confidence,

            "aspects": aspects,

            "ai_insights": ai_insights

        })


    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ==========================================
# Start Server
# ==========================================

if __name__ == "__main__":

    app.run(
    host="0.0.0.0",
    port=5000,
    debug=False
)