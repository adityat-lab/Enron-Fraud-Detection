
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "engineered_enron_data.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("Loading engineered dataset...")

# Load dataset
df = pd.read_csv(DATA_PATH, index_col="person")

# Separate features and target
X = df.drop(columns=["poi"])
y = df["poi"].astype(int)

# Ensure all input features are numeric
X = X.apply(pd.to_numeric, errors="coerce")

print("\nDataset shape:", X.shape)
print("POI records:", y.sum())
print("Non-POI records:", (y == 0).sum())

# Define machine learning algorithms
models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        max_depth=4,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        max_depth=5,
        random_state=42
    )
}

# Stratified cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1"
}

results = {}

print("\nStarting model evaluation...")

for name, classifier in models.items():

    # Pipeline prevents data leakage during imputation
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", classifier)
    ])

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        error_score="raise"
    )

    results[name] = {
        "pipeline": pipeline,
        "accuracy": scores["test_accuracy"].mean(),
        "precision": scores["test_precision"].mean(),
        "recall": scores["test_recall"].mean(),
        "f1": scores["test_f1"].mean()
    }

    print(f"\n{name}")
    print("Accuracy :", round(results[name]["accuracy"], 3))
    print("Precision:", round(results[name]["precision"], 3))
    print("Recall   :", round(results[name]["recall"], 3))
    print("F1 Score :", round(results[name]["f1"], 3))

# Compare models
results_df = pd.DataFrame({
    name: {
        metric: values[metric]
        for metric in ["accuracy", "precision", "recall", "f1"]
    }
    for name, values in results.items()
}).T

print("\nMODEL COMPARISON")
print(results_df.round(3))

# Select best model using mean cross-validation F1
best_model_name = max(
    results,
    key=lambda name: results[name]["f1"]
)

best_pipeline = results[best_model_name]["pipeline"]

# Fit selected model on the full dataset
best_pipeline.fit(X, y)

# Save trained model
model_path = MODEL_DIR / "best_model.joblib"
joblib.dump(best_pipeline, model_path)

# Save feature names
feature_path = MODEL_DIR / "feature_names.joblib"
joblib.dump(X.columns.tolist(), feature_path)

print("\nBest Model:", best_model_name)
print("Model saved successfully!")
print("Model path:", model_path)
