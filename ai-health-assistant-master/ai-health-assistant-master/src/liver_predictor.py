import joblib
import pandas as pd

model = joblib.load("models/best_liver_model.pkl")

def predict_liver(input_data):
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]

    if prediction == 1:
        return "Liver Disease Detected ⚠️"
    else:
        return "No Liver Disease ✅"