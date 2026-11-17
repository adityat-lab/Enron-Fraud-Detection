
import pandas as pd
import numpy as np
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data" / "cleaned_enron_data.csv"
OUTPUT_PATH = BASE_DIR / "data" / "engineered_enron_data.csv"

print("Starting Feature Engineering...")

# Load cleaned dataset
df = pd.read_csv(INPUT_PATH, index_col="person")

# Convert POI column to boolean
df["poi"] = df["poi"].astype(bool)

# Helper function for safe division
def safe_divide(numerator, denominator):
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator

# Feature 1: Fraction of emails received from POIs
df["fraction_from_poi"] = safe_divide(
    df["from_poi_to_this_person"],
    df["to_messages"]
)

# Feature 2: Fraction of emails sent to POIs
df["fraction_to_poi"] = safe_divide(
    df["from_this_person_to_poi"],
    df["from_messages"]
)

# Feature 3: Total compensation
compensation_columns = [
    "salary",
    "bonus",
    "long_term_incentive"
]

df["total_compensation"] = df[compensation_columns].sum(
    axis=1,
    min_count=1
)

# Feature 4: Stock option ratio
df["stock_ratio"] = safe_divide(
    df["exercised_stock_options"],
    df["total_stock_value"]
)

# Display newly created features
new_features = [
    "fraction_from_poi",
    "fraction_to_poi",
    "total_compensation",
    "stock_ratio"
]

print("\nNew Features Created:")
print(df[new_features].head(10))

# Save engineered dataset
df.to_csv(OUTPUT_PATH, index_label="person")

print("\nFeature Engineering Completed Successfully!")
print("Dataset shape:", df.shape)
print("Saved to:", OUTPUT_PATH)
