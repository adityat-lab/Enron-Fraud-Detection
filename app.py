
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
FEATURE_PATH = BASE_DIR / "models" / "feature_names.joblib"
DATA_PATH = BASE_DIR / "data" / "engineered_enron_data.csv"


# --------------------------------------------------
# 2. STREAMLIT PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="Enron Fraud Detection System",
    page_icon="🔍",
    layout="wide"
)


# --------------------------------------------------
# 3. LOAD TRAINED MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURE_PATH)
    return model, features


# --------------------------------------------------
# 4. LOAD DEFAULT FEATURE VALUES
# --------------------------------------------------

@st.cache_data
def load_defaults():
    df = pd.read_csv(DATA_PATH)

    defaults = df.drop(
        columns=["person", "poi"],
        errors="ignore"
    ).median(numeric_only=True)

    return defaults


# --------------------------------------------------
# 5. APPLICATION TITLE
# --------------------------------------------------

st.title("🔍 Enron Fraud Detection System")

st.markdown("""
### Machine Learning Based Person of Interest Detection

This application uses Machine Learning to identify individuals
who may be classified as Persons of Interest (POI) based on
financial and email communication information from the Enron dataset.
""")

st.info("""
Educational demonstration only. A POI prediction is not proof
that a person committed fraud or any other wrongdoing.
""")


# --------------------------------------------------
# 6. CHECK REQUIRED FILES
# --------------------------------------------------

if not MODEL_PATH.exists():
    st.error("Trained model not found. Please run src/train_model.py first.")
    st.stop()

if not FEATURE_PATH.exists():
    st.error("Feature names file not found. Please run src/train_model.py first.")
    st.stop()

if not DATA_PATH.exists():
    st.error("Engineered dataset not found. Please run src/feature_engineering.py first.")
    st.stop()


# --------------------------------------------------
# 7. LOAD MODEL AND DATA
# --------------------------------------------------

try:
    model, feature_names = load_model()
    defaults = load_defaults()

except Exception as error:
    st.error(f"Error loading model or dataset: {error}")
    st.stop()


# --------------------------------------------------
# 8. INPUT FORM
# --------------------------------------------------

st.divider()

st.subheader("Enter Individual Information")

st.write("""
Enter financial and email-related details below.
Default values are initialized using dataset median values.
You can change them to test hypothetical profiles.
""")

with st.form("prediction_form"):

    input_values = {}

    left_column, right_column = st.columns(2)

    for index, feature in enumerate(feature_names):

        default_value = defaults.get(feature, 0)

        if pd.isna(default_value) or not np.isfinite(default_value):
            default_value = 0.0

        default_value = float(default_value)

        target_column = (
            left_column if index % 2 == 0 else right_column
        )

        with target_column:

            input_values[feature] = st.number_input(
                label=feature.replace("_", " ").title(),
                value=default_value,
                step=1.0,
                format="%.4f",
                help="Enter a value for this feature."
            )

    submitted = st.form_submit_button(
        "Predict Person of Interest",
        use_container_width=True
    )


# --------------------------------------------------
# 9. MAKE PREDICTION
# --------------------------------------------------

if submitted:

    input_df = pd.DataFrame(
        [input_values],
        columns=feature_names
    )

    try:
        prediction = model.predict(input_df)[0]

        st.divider()
        st.subheader("Prediction Result")

        if prediction == 1:

            st.error("⚠️ Model Prediction: POI")

            st.write("""
            The model classifies this hypothetical profile
            as a Person of Interest.
            """)

        else:

            st.success("✅ Model Prediction: Non-POI")

            st.write("""
            The model classifies this hypothetical profile
            as a Non-Person of Interest.
            """)

        # Show prediction probability
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_df)[0]

            classes = list(model.classes_)

            if 1 in classes:

                poi_probability = probabilities[classes.index(1)]

                st.metric(
                    "Model POI Probability",
                    f"{poi_probability * 100:.2f}%"
                )

                st.progress(float(poi_probability))

    except Exception as error:

        st.error(f"Prediction failed: {error}")


# --------------------------------------------------
# 10. FOOTER
# --------------------------------------------------

st.divider()

st.caption("""
Developed as an educational Machine Learning project
using the Enron dataset.
""")
