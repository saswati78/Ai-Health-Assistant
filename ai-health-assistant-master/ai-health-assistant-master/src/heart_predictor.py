import joblib
import pandas as pd

# Load model & scaler
heart_model = joblib.load("models/best_heart_model.pkl")
heart_scaler = joblib.load("models/scaler.pkl")

# Exact feature order used during training
FEATURE_ORDER = [
    'Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',
    'Sex_M',
    'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA',
    'RestingECG_Normal', 'RestingECG_ST',
    'ExerciseAngina_Y',
    'ST_Slope_Flat', 'ST_Slope_Up'
]


def predict_heart(data):

    # Convert to DataFrame
    df = pd.DataFrame([data])

    # Fix missing columns + correct order
    df = df.reindex(columns=FEATURE_ORDER, fill_value=0)

    # Scale
    scaled = heart_scaler.transform(df)

    # Predict
    prediction = heart_model.predict(scaled)[0]

    return "Heart Disease Risk Detected" if prediction == 1 else "No Heart Disease Risk"