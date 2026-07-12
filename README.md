# 🏥 AI Health Assistant

AI-powered health prediction platform using Machine Learning (ML) and Deep Learning (DL) models to detect heart disease, diabetes, liver disease, breast cancer, and brain tumor risks. The system also provides personalized lifestyle recommendations using Google's Gemini AI.

An integrated AI health prediction system that leverages **Machine Learning** and **Deep Learning** models to analyze clinical data and MRI scans, then converts each prediction into clear, actionable lifestyle guidance using the **Google Gemini API**.

---

## 🌟 Features

- ❤️ **Heart Disease Prediction** — Analyzes 15 clinical features to detect potential heart disease risk.
- 🩸 **Diabetes Prediction** — Uses 8 health metrics to assess diabetes risk.
- 🧬 **Liver Disease Prediction** — Evaluates liver health using 10 biochemical parameters.
- 🎗️ **Breast Cancer Detection** — Classifies tumors as **Malignant** or **Benign**.
- 🧠 **Brain Tumor Detection** — Uses an EfficientNetB0 Deep Learning model to analyze MRI images.
- 🤖 **AI Lifestyle Advisor** — Generates personalized diet, exercise, and prevention advice using the Google Gemini API.

---

## 💡 Highlights

- 🔍 Combines classical Machine Learning models (Random Forest, SVM) with Deep Learning for comprehensive disease prediction.
- ⚙️ Modular Flask backend with separate predictor modules for each disease.
- 📊 Supports multiple clinical datasets and MRI image analysis.
- 🖥️ Lightweight frontend built using Jinja2 templates and Vanilla CSS.
- 🔐 Secure API key management using a `.env` file.

---

## 🛠️ Technology Stack

| Layer | Technologies |
|-------|--------------|
| **Backend** | Flask (Python) |
| **Machine Learning** | Scikit-learn, Joblib |
| **Deep Learning** | TensorFlow, Keras, OpenCV (EfficientNetB0) |
| **AI Integration** | Google Gemini API |
| **Data Processing** | Pandas, NumPy |
| **Frontend** | HTML, Jinja2, CSS |

---

## 📁 Project Structure

```text
Ai-Health-Assistant/
│
├── app.py
├── src/
│   ├── brain_tumor_predictor.py
│   ├── breast_predictor.py
│   ├── diabetes_predictor.py
│   ├── heart_predictor.py
│   ├── liver_predictor.py
│   └── lifestyle_llm.py
│
├── models/
├── static/
├── templates/
├── notebook/
├── requirements.txt
├── feature.py
└── README.md
```

---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/saswati78/Ai-Health-Assistant.git
cd Ai-Health-Assistant
```

### 2. Create a Virtual Environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root and add your Gemini API key:

```env
GEMINI_API_KEY=your_api_key_here
```

### 5. Run the Application

```bash
python app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000
```

---

## 📌 Future Enhancements

- Add user authentication
- Store prediction history
- Deploy using Docker
- Improve model accuracy
- Add more disease prediction modules
- Responsive mobile interface

---

