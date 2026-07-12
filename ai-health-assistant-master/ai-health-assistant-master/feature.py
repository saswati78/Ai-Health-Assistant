import joblib
import pandas as pd

# Load trained pipeline
breast_pipeline = joblib.load("models/breast_cancer_pipeline.pkl")


print(breast_pipeline.feature_names_in_)