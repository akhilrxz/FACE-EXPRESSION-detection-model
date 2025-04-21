import streamlit as st
import cv2
import numpy as np
from collections import deque, Counter
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import json

# === Load model and class labels ===
model = load_model("expression_model.h5")

with open("class_labels.json", "r") as f:
    class_indices = json.load(f)
classes = [None] * len(class_indices)
for label, index in class_indices.items():
    classes[index] = label

# === Initialize prediction buffer for smoothing ===
prediction_history = deque(maxlen=10)

# === Load face detector ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# === Streamlit App ===
st.title("😄 Real-Time Facial Expression Detector")
run = st.checkbox('Start Webcam')

FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    if not ret:
        st.warning("Camera not detected")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_input = cv2.resize(face, (96, 96))
        face_input = img_to_array(face_input) / 255.0
        face_input = np.expand_dims(face_input, axis=0)

        preds = model.predict(face_input, verbose=0)[0]
        raw_pred = classes[np.argmax(preds)]
        prediction_history.append(raw_pred)

        # Use most frequent label in last 10 predictions
        smoothed_label = Counter(prediction_history).most_common(1)[0][0]
        confidence = np.max(preds) * 100

        # Draw rectangle & label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{smoothed_label} ({confidence:.1f}%)", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)

camera.release()
