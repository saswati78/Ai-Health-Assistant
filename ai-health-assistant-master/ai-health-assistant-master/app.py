from flask import Flask, render_template, request
import json, re, random
import os

from src.brain_tumor_predictor import predict_brain
from src.heart_predictor import predict_heart
from src.diabetes_predictor import predict_diabetes
from src.liver_predictor import predict_liver
from src.breast_predictor import predict_breast
from src.lifestyle_llm import get_lifestyle_advice

app = Flask(__name__)

# -------------------------
# Clean JSON from LLM
# -------------------------
def clean_json(text):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    return match.group() if match else "{}"

# -------------------------
# Helper: Generate severity
# -------------------------
def generate_severity(result, advice_dict):
    if isinstance(result, float):
        return result * 100

    is_dangerous = advice_dict.get("is_dangerous", False)

    if is_dangerous:
        return random.randint(50, 100)   # dangerous → high severity
    else:
        return random.randint(0, 50)     # safe → low severity

# -------------------------
# Home
# -------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -------------------------
# Pages
# -------------------------
@app.route("/heart")
def heart():
    return render_template("heart.html")

@app.route("/diabetes")
def diabetes():
    return render_template("diabates.html")

@app.route("/liver")
def liver():
    return render_template("liver.html")

@app.route("/breast")
def breast():
    return render_template("breast.html")

@app.route("/brain")
def brain():
    return render_template("brain.html")

# -------------------------
# Heart
# -------------------------
@app.route("/predict-heart", methods=["POST"])
def heart_result():
    data = {key: float(request.form[key]) for key in request.form}
    result = predict_heart(data)

    advice_text = get_lifestyle_advice(result)
    advice_dict = json.loads(clean_json(advice_text))

    severity = generate_severity(result, advice_dict)

    return render_template("result.html",
                           prediction=result,
                           advice=advice_dict,
                           severity=severity)

# -------------------------
# Diabetes
# -------------------------
@app.route("/predict-diabetes", methods=["POST"])
def diabetes_result():
    data = {key: float(request.form[key]) for key in request.form}
    result = predict_diabetes(data)

    advice_text = get_lifestyle_advice(result)
    advice_dict = json.loads(clean_json(advice_text))

    severity = generate_severity(result, advice_dict)

    return render_template("result.html",
                           prediction=result,
                           advice=advice_dict,
                           severity=severity)

# -------------------------
# Liver
# -------------------------
@app.route("/predict-liver", methods=["POST"])
def liver_result():
    data = {key: float(request.form[key]) for key in request.form}
    result = predict_liver(data)

    advice_text = get_lifestyle_advice(result)
    advice_dict = json.loads(clean_json(advice_text))

    severity = generate_severity(result, advice_dict)

    return render_template("result.html",
                           prediction=result,
                           advice=advice_dict,
                           severity=severity)

# -------------------------
# Breast
# -------------------------
@app.route("/predict-breast", methods=["POST"])
def breast_result():

    feature_names = [
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

    try:
        data = {f: float(request.form[f]) for f in feature_names}
    except:
        return "Invalid input", 400

    result = predict_breast(data)

    advice_text = get_lifestyle_advice(result["label"])
    advice_dict = json.loads(clean_json(advice_text))

    severity = result["prob_malignant"] * 100

    return render_template(
        "result.html",
        prediction=result["label"],
        probability=round(result["prob_malignant"] * 100, 2),
        confidence=round(result["confidence"] * 100, 2),
        advice=advice_dict,
        severity=severity
    )

# -------------------------
# Brain
# -------------------------
@app.route("/predict-brain", methods=["POST"])
def brain_result():
    file = request.files['file']
    
    path = os.path.join("static", file.filename)
    file.save(path)

    label, confidence = predict_brain(path)

    # Treat tumor cases as dangerous
    is_dangerous = label != "notumor"

    advice_text = get_lifestyle_advice(label)
    advice_dict = json.loads(clean_json(advice_text))

    severity = confidence if is_dangerous else confidence * 0.3

    return render_template("result.html",
                           prediction=label,
                           confidence=round(confidence, 2),
                           advice=advice_dict,
                           severity=severity,
                           image_path=path)

# -------------------------
if __name__ == "__main__":
    app.run(debug=True)