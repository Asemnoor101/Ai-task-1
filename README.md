# Ai-task-1

# Teachable Machine - Image Classification Project

This project is an **image classification system** trained using **Google Teachable Machine** and deployed using **TensorFlow & OpenCV**. The model recognizes different hand gestures in real-time using a webcam.

---

## 🚀 **Project Overview**
1. **Trained a Model** using **Google Teachable Machine**.
2. **Created three classes**:
   - ✋ `Class 1`: Open Hand
   - 👍 `Class 2`: LIke sign
   - ✌ `Class 4`: Peace Sign
3. **Exported the model** in TensorFlow **(.h5 format)**.
4. **Developed a Python script** to load the model and perform real-time predictions using a webcam.
5. **Implemented OpenCV** for capturing webcam frames and displaying predictions.

---

## 📸 **Model Training & Export**
### **Model Training in Teachable Machine**
![لقطة شاشة 2025-01-31 022845](https://github.com/user-attachments/assets/62b11912-596e-42c4-bc8c-b0aaf6c9e93a)


### **Exporting the Model as `.h5`**
![لقطة شاشة 2025-01-31 022304](https://github.com/user-attachments/assets/ded09a9e-2a50-41c2-82fa-e374b17a1297)


---

## Live Prediction Script
This script loads the trained model and captures real-time predictions from a webcam.

```python
import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

model_path = os.path.abspath("model_saved/model.h5")
labels_path = os.path.abspath("model_saved/labels.txt")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

if not os.path.exists(labels_path):
    raise FileNotFoundError(f"Labels file not found at: {labels_path}")

model = tf.keras.models.load_model(model_path, compile=False)

with open(labels_path, "r") as f:
    labels = [line.strip() for line in f.readlines()]

def preprocess_frame(frame):
    size = (224, 224)
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    return np.expand_dims(normalized_image_array, axis=0)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image_data = preprocess_frame(frame)
    prediction = model.predict(image_data)
    index = np.argmax(prediction)
    class_name = labels[index]
    confidence_score = prediction[0][index]

    text = f"{class_name}: {confidence_score:.2f}"
    cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Live Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


```

