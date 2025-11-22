# train_model.py
# ===============================================
# 🧠 Disease Prediction using Neural Network
# ===============================================

import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import numpy as np

# -------------------------------------------------
# 1️⃣ Load dataset
# -------------------------------------------------
try:
    df = pd.read_csv("DiseaseDataset.csv")
    print("✅ Loaded dataset from CSV file")
except FileNotFoundError:
    print("⚠️ CSV file not found, please place 'DiseaseDataset.csv' in project folder.")
    raise SystemExit("❌ Training stopped.")

# -------------------------------------------------
# 2️⃣ Prepare features and target
# -------------------------------------------------
X = df.drop(columns=["diseases"])
y = df["diseases"]

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)
y_categorical = to_categorical(y_encoded, num_classes)

# -------------------------------------------------
# 3️⃣ Split into train and test sets
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42
)
print(f"📊 Training samples: {X_train.shape[0]} | Testing samples: {X_test.shape[0]}")

# -------------------------------------------------
# 4️⃣ Build Neural Network model
# -------------------------------------------------
model = Sequential([
    Dense(512, input_dim=X.shape[1], activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dense(num_classes, activation='softmax')
])

# -------------------------------------------------
# 5️⃣ Compile and train
# -------------------------------------------------
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=40,
    batch_size=256,
    callbacks=[early_stop],
    verbose=1
)

# -------------------------------------------------
# 6️⃣ Evaluate model
# -------------------------------------------------
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ Model trained with accuracy: {acc * 100:.2f}%")

# -------------------------------------------------
# 7️⃣ Save model and encoder
# -------------------------------------------------
model.save("disease_nn_model.h5")
joblib.dump(label_encoder, "label_encoder.pkl")

print("💾 Model saved as 'disease_nn_model.h5'")
print("💾 Label encoder saved as 'label_encoder.pkl'")

# -------------------------------------------------
# 8️⃣ Example prediction
# -------------------------------------------------
new_sample = np.zeros((1, X.shape[1]))
new_sample[0, 5] = 1
new_sample[0, 20] = 1
new_sample[0, 150] = 1

pred = model.predict(new_sample)
predicted_class = np.argmax(pred)
predicted_disease = label_encoder.inverse_transform([predicted_class])[0]

print(f"🩺 Predicted Disease: {predicted_disease}")
