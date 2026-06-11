import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ==========================================
# Create Results Directory
# ==========================================

os.makedirs(
    "results/mobilenet",
    exist_ok=True
)


# ==========================================
# Load Model
# ==========================================

model = load_model(
    "results/mobilenet/mobilenet.keras"
)


# ==========================================
# Validation Dataset (shuffle=True — matches training split)
# ==========================================

val_ds = image_dataset_from_directory(
    "data/trashNet-dataset",
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(224, 224),
    batch_size=32,
    shuffle=True
)

class_names = val_ds.class_names


# ==========================================
# Single-pass collection of images and labels
# ==========================================

all_images = []
all_labels = []

for x_batch, y_batch in val_ds:
    all_images.append(x_batch.numpy())
    all_labels.append(y_batch.numpy())

X_raw = np.concatenate(all_images, axis=0)
y_true = np.concatenate(all_labels, axis=0).astype(int)

X = preprocess_input(X_raw)


# ==========================================
# Predictions (on numpy array — no re-iteration)
# ==========================================

predictions = model.predict(X, batch_size=32, verbose=1)

y_pred = np.argmax(predictions, axis=1)


# ==========================================
# Classification Report
# ==========================================

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names
)

print("\n==============================")
print("MobileNetV2 — Classification Report")
print("==============================\n")
print(report)

with open("results/mobilenet/classification_report.txt", "w") as f:
    f.write("MobileNetV2 — Classification Report\n\n")
    f.write(report)


# ==========================================
# Confusion Matrix
# ==========================================

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("MobileNetV2 — Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()

plt.savefig("results/mobilenet/confusion_matrix.png")
plt.close()

print("Confusion matrix saved.")
print("Classification report saved.")
