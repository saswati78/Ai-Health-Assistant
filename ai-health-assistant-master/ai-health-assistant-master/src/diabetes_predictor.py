import joblib
import pandas as pd

diabetes_model = joblib.load("models/Diabetes_Model.pkl")
diabetes_preprocessor = joblib.load("models/Diabetes_Preprocessor.pkl")


def predict_diabetes(data):

    df = pd.DataFrame([data])

    processed = diabetes_preprocessor.transform(df)

    prediction = diabetes_model.predict(processed)[0]

    if prediction == 1:
        return "Diabetes Risk Detected"
    else:
        return "No Diabetes Risk"