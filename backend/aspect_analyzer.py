import re


# Keywords associated with different customer-feedback aspects
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


def find_aspects(feedback):

    feedback = feedback.lower()

    detected_aspects = []

    for aspect, keywords in ASPECT_KEYWORDS.items():

        for keyword in keywords:

            if keyword in feedback:
                detected_aspects.append(aspect)
                break

    return detected_aspects


# Test the analyzer
feedback = input("\nEnter customer feedback: ")

aspects = find_aspects(feedback)

print("\nDetected Aspects:")

if aspects:

    for aspect in aspects:
        print("-", aspect)

else:
    print("No specific aspect detected.")