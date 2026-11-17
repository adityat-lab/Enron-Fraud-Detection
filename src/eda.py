
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "cleaned_enron_data.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "graphs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load cleaned data
df = pd.read_csv(DATA_PATH, index_col="person")

# Convert POI to readable labels
df["poi_label"] = df["poi"].map({
    True: "POI",
    False: "Non-POI"
})

sns.set_theme(style="whitegrid")

# 1. POI distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="poi_label")
plt.title("POI vs Non-POI Distribution")
plt.xlabel("Person Category")
plt.ylabel("Number of People")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "poi_distribution.png")
plt.close()

# 2. Salary distribution
plt.figure(figsize=(8, 5))
sns.histplot(
    data=df,
    x="salary",
    hue="poi_label",
    bins=25,
    kde=True
)
plt.title("Salary Distribution")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "salary_distribution.png")
plt.close()

# 3. Bonus distribution
plt.figure(figsize=(8, 5))
sns.histplot(
    data=df,
    x="bonus",
    hue="poi_label",
    bins=25
)
plt.title("Bonus Distribution")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "bonus_distribution.png")
plt.close()

# 4. Total stock value comparison
plt.figure(figsize=(8, 5))
sns.boxplot(
    data=df,
    x="poi_label",
    y="total_stock_value"
)
plt.title("Total Stock Value: POI vs Non-POI")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "stock_value_comparison.png")
plt.close()

# 5. Correlation heatmap
numeric_df = df.drop(columns=["poi_label", "poi"])

plt.figure(figsize=(15, 10))
sns.heatmap(
    numeric_df.corr(),
    cmap="coolwarm",
    center=0,
    annot=False
)
plt.title("Financial Features Correlation Heatmap")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_heatmap.png")
plt.close()

print("EDA completed successfully!")
print("Graphs saved in:", OUTPUT_DIR)

print("\nBasic statistical summary:")
print(df.describe())
