import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load data
X = np.load("data/landmarks.npy")
y = np.load("data/labels.npy")

# -----------------------------
# ✅ STEP 1: Normalize landmarks
# -----------------------------
X = X.reshape(-1, 21, 3)

# Center around wrist (landmark 0)
wrist = X[:, 0:1, :]
X = X - wrist

# Scale normalization
scale = np.linalg.norm(X, axis=(1, 2), keepdims=True) + 1e-6
X = X / scale

# Reshape back
X = X.reshape(-1, 63)

# -----------------------------
# ✅ STEP 2: Data Augmentation
# -----------------------------
def augment(X, y):
    noise = np.random.normal(0, 0.01, X.shape)
    X_noise = X + noise

    scale = np.random.uniform(0.9, 1.1, (X.shape[0], 1))
    X_scaled = X * scale

    X_aug = np.vstack((X, X_noise, X_scaled))
    y_aug = np.hstack((y, y, y))

    return X_aug, y_aug

X, y = augment(X, y)

# -----------------------------
# ✅ STEP 3: Train/Val/Test Split
# -----------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5,  random_state=42
)

# Reshape for CNN
X_train = X_train.reshape(-1, 63, 1)
X_val = X_val.reshape(-1, 63, 1)
X_test = X_test.reshape(-1, 63, 1)

# -----------------------------
# ✅ STEP 4: Model (Regularized)
# -----------------------------
model = Sequential([
    Conv1D(32, 3, activation='relu',
           kernel_regularizer=regularizers.l2(1e-4),
           input_shape=(63, 1)),
    Dropout(0.3),

    Conv1D(64, 3, activation='relu',
           kernel_regularizer=regularizers.l2(1e-4)),
    Dropout(0.4),

    Flatten(),

    Dense(64, activation='relu',
          kernel_regularizer=regularizers.l2(1e-4)),
    Dropout(0.5),

    Dense(5, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# ✅ STEP 5: Early Stopping
# -----------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# -----------------------------
# ✅ STEP 6: Train
# -----------------------------
history = model.fit(
    X_train, y_train,
    epochs=20,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    batch_size=32,
    shuffle=True
)

# -----------------------------
# ✅ STEP 7: Evaluation
# -----------------------------
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# ✅ STEP 8: Save Model
# -----------------------------
os.makedirs("models", exist_ok=True)
model.save("models/gesture_model.h5")

print("✅ Model Trained & Saved Successfully!")