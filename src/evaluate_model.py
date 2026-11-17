
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    balanced_accuracy_score
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "engineered_enron_data.csv"
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
FEATURE_PATH = BASE_DIR / "models" / "feature_names.joblib"

OUTPUT_DIR = BASE_DIR / "outputs" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading dataset and trained model...")

# Load dataset
df = pd.read_csv(DATA_PATH, index_col="person")

X = df.drop(columns=["poi"])
y = df["poi"].astype(int)

# Load trained pipeline and feature names
model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURE_PATH)

# Maintain original training feature order
X = X[feature_names]

# Five-fold stratified cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("\nGenerating out-of-fold predictions...")

# Each person is predicted by a model that did not train on that person's fold.
y_pred = cross_val_predict(
    model,
    X,
    y,
    cv=cv,
    method="predict"
)

# Calculate evaluation metrics
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred, zero_division=0)
recall = recall_score(y, y_pred, zero_division=0)
f1 = f1_score(y, y_pred, zero_division=0)
balanced_acc = balanced_accuracy_score(y, y_pred)

print("\nMODEL EVALUATION RESULTS")
print("------------------------")

print("Accuracy:", round(accuracy, 3))
print("Precision:", round(precision, 3))
print("Recall:", round(recall, 3))
print("F1 Score:", round(f1, 3))
print("Balanced Accuracy:", round(balanced_acc, 3))

# Classification report
report = classification_report(
    y,
    y_pred,
    target_names=["Non-POI", "POI"],
    zero_division=0
)

print("\nCLASSIFICATION REPORT")
print(report)

# Save report
with open(OUTPUT_DIR / "classification_report.txt", "w") as file:
    file.write(report)

# Confusion matrix
cm = confusion_matrix(y, y_pred, labels=[0, 1])

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Non-POI", "POI"],
    yticklabels=["Non-POI", "POI"]
)

plt.title("Fraud Detection - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "confusion_matrix.png")
plt.close()

# Save predictions
predictions = pd.DataFrame({
    "Actual_POI": y,
    "Predicted_POI": y_pred
}, index=X.index)

predictions.to_csv(OUTPUT_DIR / "predictions.csv")

print("\nEvaluation completed successfully!")
print("Results saved in:", OUTPUT_DIR)
