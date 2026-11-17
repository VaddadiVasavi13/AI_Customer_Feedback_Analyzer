import pandas as pd

# Load the customer feedback dataset
data = pd.read_csv("../dataset/customer_feedback.csv")

# Display the first 5 rows
print("First 5 customer feedback records:")
print(data.head())

# Display dataset information
print("\nDataset information:")
print(data.info())

# Count each sentiment
print("\nSentiment distribution:")
print(data["sentiment"].value_counts())