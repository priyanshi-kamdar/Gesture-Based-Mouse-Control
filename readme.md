# 🖐️ Gesture-Based Mouse Control 

## 📌 Project Overview

This project implements a **real-time gesture-based mouse control system** using **Mediapipe** for hand landmark extraction and a **Convolutional Neural Network (CNN)** trained on landmark data.

The system captures hand gestures via webcam and maps them to mouse actions such as scrolling, clicking, and taking screenshots.

---

## 🎯 Objectives

* Build a real-time gesture recognition system
* Use **Mediapipe** to extract hand landmarks (21 keypoints)
* Train a **CNN model on landmark data (not raw images)**
* Map gestures to system-level mouse controls

---

## 🧠 How It Works

### 🔹 Step 1: Landmark Extraction

* Mediapipe detects hand and extracts **21 landmarks**
* Each landmark has (x, y, z) → total **63 features**

### 🔹 Step 2: Data Preprocessing

* Images → landmarks
* Normalize and store as `.npy` files
* Label encoding for gestures

### 🔹 Step 3: Model Training

* CNN is trained on landmark vectors
* Learns patterns for different gestures

### 🔹 Step 4: Real-Time Prediction

* Webcam captures live feed
* Mediapipe extracts landmarks
* CNN predicts gesture
* Action is triggered using PyAutoGUI

---

## 🎮 Gesture Controls

| Gesture        | Action      |
| -------------- | ----------- |
| ✋ Open Palm    | Screenshot  |
| ✌️ Peace Sign  | Right Click |
| 👍 Thumbs Up   | Scroll Up   |
| 👎 Thumbs Down | Scroll Down |
| 👆 Pointing    | Mouse Movement |

---

## 📁 Project Structure

```
gesture-mouse-control/
│
├── dataset/
│   ├── open_palm/
│   ├── peace/
│   ├── thumbs_up/
│   └── thumbs_down/
│   └── pointing/
│
├── data/
│   ├── landmarks.npy
│   └── labels.npy
│
├── models/
│   └── gesture_model.h5
│
├── preprocess.py
├── train.py
├── predict.py
├── utils.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Create Environment

```bash
conda create -n gesture_env python=3.10
conda activate gesture_env
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Execution Steps

### ▶️ Step 1: Preprocess Dataset

Extract landmarks from dataset images:

```bash
python preprocess.py
```

Outputs:

* `data/landmarks.npy`
* `data/labels.npy`

---

### ▶️ Step 2: Train Model

Train CNN on landmark data:

```bash
python train.py
```

Output:

* `models/gesture_model.h5`

---

### ▶️ Step 3: Run Real-Time System

```bash
python predict.py
```

---

## ⚡ Features

* ✅ Real-time gesture detection
* ✅ CNN trained on landmark data
* ✅ Smooth predictions using buffering
* ✅ Cooldown mechanism to prevent repeated actions
* ✅ Lightweight and fast inference

---

## 📊 Model Details

* Input: 63-dimensional landmark vector
* Architecture: 1D CNN
* Loss Function: Sparse Categorical Crossentropy
* Optimizer: Adam
* Output: 5 gesture classes


---

## ⚠️ Important Notes

* Use **Python 3.10** (required for Mediapipe compatibility)
* Ensure dataset folder structure is correct
* Good lighting improves detection accuracy
* Avoid background noise for better performance

---

## 🚀 Future Improvements

* Add more gestures (drag, zoom, etc.)
* Build GUI dashboard
* Deploy as desktop application
* Improve accuracy using LSTM or hybrid models

---

## 🏆 Tech Stack

* Python
* OpenCV
* Mediapipe
* TensorFlow / Keras
* NumPy
* PyAutoGUI



## 📌 Summary

This project demonstrates how **computer vision + deep learning** can be used to create intuitive human-computer interaction systems, replacing traditional input devices with natural gestures.
