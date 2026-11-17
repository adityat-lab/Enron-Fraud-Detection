
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "final_project_dataset.pkl"
OUTPUT_PATH = BASE_DIR / "data" / "cleaned_enron_data.csv"

print("Starting data cleaning...")

# Load original dataset
with open(DATA_PATH, "rb") as file:
    data = pickle.load(file)

df = pd.DataFrame.from_dict(data, orient="index")

# Convert missing-value strings to actual NaN
df.replace(["NaN", "nan", ""], np.nan, inplace=True)

print("\nOriginal dataset shape:", df.shape)

# Step 1: Remove non-person / summary records
records_to_remove = [
    "TOTAL",
    "THE TRAVEL AGENCY IN THE PARK"
]

df.drop(
    index=records_to_remove,
    errors="ignore",
    inplace=True
)

# Step 2: Remove records with no useful information
# Keep the POI label, but check whether all other fields are missing.
feature_columns = [col for col in df.columns if col != "poi"]

empty_records = df[feature_columns].isnull().all(axis=1)

print("\nCompletely empty records:")
print(df.index[empty_records].tolist())

df = df.loc[~empty_records].copy()

# Step 3: Remove email address
df.drop(columns=["email_address"], inplace=True)

# Step 4: Convert feature columns to numeric
for column in df.columns:
    if column != "poi":
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

# Step 5: Check duplicate records
print("\nDuplicate rows:", df.duplicated().sum())

# Step 6: Check missing values
print("\nMissing values after cleaning:")
print(df.isnull().sum())

# Step 7: Check target distribution
print("\nPOI distribution:")
print(df["poi"].value_counts())

# Step 8: Save cleaned dataset
df.to_csv(OUTPUT_PATH, index_label="person")

print("\nCleaned dataset shape:", df.shape)
print("\nData cleaning completed successfully!")
print("Saved file:", OUTPUT_PATH)
