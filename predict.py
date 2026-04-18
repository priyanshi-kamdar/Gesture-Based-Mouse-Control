import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from utils import get_stable_prediction, perform_action

# Load trained model
model = load_model("models/gesture_model.h5")

mp_hands = mp.solutions.hands.Hands()
cap = cv2.VideoCapture(0)

def extract_landmarks(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = mp_hands.process(rgb)

    if result.multi_hand_landmarks:
        landmarks = result.multi_hand_landmarks[0]

        data = []
        for lm in landmarks.landmark:
            data.append([lm.x, lm.y, lm.z])

        return np.array(data)  # shape (21,3)
    return None


while True:
    ret, frame = cap.read()
    if not ret:
        break

    landmarks = extract_landmarks(frame)

    if landmarks is not None:
        
        # Save absolute coordinates before normalization for cursor pointing
        absolute_landmarks = np.copy(landmarks)

        # -----------------------------
        # ✅ SAME NORMALIZATION AS TRAINING
        # -----------------------------
        wrist = landmarks[0]
        landmarks = landmarks - wrist

        scale = np.linalg.norm(landmarks) + 1e-6
        landmarks = landmarks / scale

        # reshape for model
        X = landmarks.reshape(1, 63, 1)

        # prediction
        pred = np.argmax(model.predict(X, verbose=0))

        # smoothing
        stable_pred = get_stable_prediction(pred)

        # perform action
        perform_action(stable_pred, original_landmarks=absolute_landmarks)

        # label names (better UI)
        labels = ["Open Palm", "Peace", "Thumbs Up", "Thumbs Down", "Pointing"]

        cv2.putText(frame, f"Gesture: {labels[stable_pred]}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

    cv2.imshow("Gesture Control", frame)

    # ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()