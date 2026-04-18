import os
import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands.Hands(static_image_mode=True)

DATASET_PATH = "DL data"
X, y = [], []

label_map = {
    "Open Palm-samples": 0,
    "Peace Sign Samples": 1,
    "Thumb Up-samples": 2,
    "Thumb Down-samples": 3,
    "Pointing-samples": 4
}

def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = mp_hands.process(image_rgb)

    if result.multi_hand_landmarks:
        landmarks = result.multi_hand_landmarks[0]
        data = []
        
        # Make all coordinates relative to the wrist (landmark 0)
        base_x = landmarks.landmark[0].x
        base_y = landmarks.landmark[0].y
        base_z = landmarks.landmark[0].z
        
        for lm in landmarks.landmark:
            data.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
        return data
    return None

for label in os.listdir(DATASET_PATH):
    path = os.path.join(DATASET_PATH, label)

    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        image = cv2.imread(img_path)

        if image is None:
            continue

        landmarks = extract_landmarks(image)

        if landmarks:
            X.append(landmarks)
            y.append(label_map[label])

X = np.array(X)
y = np.array(y)

os.makedirs("data", exist_ok=True)
np.save("data/landmarks.npy", X)
np.save("data/labels.npy", y)

print("Preprocessing Done!")