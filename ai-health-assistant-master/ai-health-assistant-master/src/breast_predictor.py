import joblib
import pandas as pd

breast_pipeline = joblib.load("models/breast_cancer_pipeline.pkl")

# Model expects THESE names (with spaces)
MODEL_FEATURES = [
    "concave points_mean",
    "perimeter_worst",
    "concave points_worst",
    "radius_worst",
    "area_worst",
    "concavity_mean",
    "area_mean",
    "area_se",
    "perimeter_mean",
    "concavity_worst"
]

# Form uses THESE names (underscores)
FORM_FEATURES = [
    "concave_points_mean",
    "perimeter_worst",
    "concave_points_worst",
    "radius_worst",
    "area_worst",
    "concavity_mean",
    "area_mean",
    "area_se",
    "perimeter_mean",
    "concavity_worst"
]


def predict_breast(data: dict):
    # Step 1: Convert keys manually (underscore → space)
    mapped_data = {}

    for key in data:
        new_key = key.replace("_", " ")
        mapped_data[new_key] = data[key]

    # Step 2: Create DataFrame
    df = pd.DataFrame([mapped_data])

    # Step 3: Ensure ALL required features exist
    for col in MODEL_FEATURES:
        if col not in df.columns:
            df[col] = 0.0   # safe default

    # Step 4: Fix order
    df = df[MODEL_FEATURES]

    # Step 5: Predict
    probs = breast_pipeline.predict_proba(df)[0]

    prob_malignant = float(probs[1])
    prob_benign = float(probs[0])

    threshold = 0.4
    label = "Malignant" if prob_malignant > threshold else "Benign"

    return {
        "label": label,
        "prob_malignant": prob_malignant,
        "prob_benign": prob_benign,
        "confidence": max(prob_malignant, prob_benign)
    }