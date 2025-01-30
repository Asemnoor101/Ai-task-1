# Ai-task-1

# Teachable Machine - Image Classification Project

This project is an **image classification system** trained using **Google Teachable Machine** and deployed using **TensorFlow & OpenCV**. The model recognizes different hand gestures in real-time using a webcam.

---

## 🚀 **Project Overview**
1. **Trained a Model** using **Google Teachable Machine**.
2. **Created three classes**:
   - ✋ `Class 1`: Open Hand
   - 👍 `Class 2`: Thumbs Up
   - ✌ `Class 4`: Peace Sign
3. **Exported the model** in TensorFlow **(.h5 format)**.
4. **Developed a Python script** to load the model and perform real-time predictions using a webcam.
5. **Implemented OpenCV** for capturing webcam frames and displaying predictions.

---

## 📸 **Model Training & Export**
### **Model Training in Teachable Machine**
![Model Training](./لقطة شاشة 2025-01-31 005651.png)

### **Exporting the Model as `.h5`**
![Export Model](./لقطة شاشة 2025-01-31 022304.png)

---

## 🔧 **Installation & Setup**
### **1️⃣ Install Dependencies**
Run the following command to install all required libraries:
```bash
pip install tensorflow numpy opencv-python pillow h5py
