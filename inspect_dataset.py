import pandas as pd

# Load dataset
file_path = "dataset/Dataset-SA.csv"

df = pd.read_csv(file_path)

print("\n========================================")
print("DATASET INFORMATION")
print("========================================")

print("\nNumber of rows:", len(df))
print("Number of columns:", len(df.columns))

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nUnique values in each column:")
for column in df.columns:
    print("\n", column)
    print(df[column].unique()[:20])

print("\n========================================")
print("SENTIMENT DISTRIBUTION")
print("========================================")

print(df["Sentiment"].value_counts())

print("\nSentiment percentages:")
print(
    df["Sentiment"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)