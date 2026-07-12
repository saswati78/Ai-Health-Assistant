import numpy as np
import cv2
import os
from keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# Load model once
model_path = os.path.join("models", "brain_tumor_efficientnet.keras")
model = load_model(model_path)

class_labels = {
    0: "Glioma Tumor",
    1: "Meningioma Tumor",
    2: "No Tumor",
    3: "Pituitary Tumor"
}

def predict_brain(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)

    img = np.reshape(img, (1, 224, 224, 3))

    pred = model.predict(img)
    class_index = np.argmax(pred)
    confidence = float(np.max(pred)) * 100

    label = class_labels[class_index]
    return label, confidence