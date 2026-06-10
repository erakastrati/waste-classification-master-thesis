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


# ==========================================
# Create Results Directory
# ==========================================

os.makedirs(
    "results/cnn_baseline",
    exist_ok=True
)


# ==========================================
# Load Model
# ==========================================

model = load_model(
    "results/cnn_baseline/cnn_baseline.keras"
)


# ==========================================
# Validation Dataset
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

val_ds = val_ds.map(
    lambda x, y: (x / 255.0, y)
)


# ==========================================
# Predictions
# ==========================================

y_true = np.concatenate(
    [y.numpy() for _, y in val_ds]
)

print("\nClass Names:")
print(class_names)

print("\nUnique labels in y_true:")
print(np.unique(y_true))

print("\nNumber of samples:")
print(len(y_true))

print("\nLabel counts:")
unique, counts = np.unique(y_true, return_counts=True)

for label, count in zip(unique, counts):
    print(f"Label {label}: {count}")

predictions = model.predict(
    val_ds
)

y_pred = np.argmax(
    predictions,
    axis=1
)


# ==========================================
# Classification Report
# ==========================================

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names
)

print("\nClassification Report:\n")
print(report)

with open(
    "results/cnn_baseline/classification_report.txt",
    "w"
) as f:
    f.write(report)


# ==========================================
# Confusion Matrix
# ==========================================

cm = confusion_matrix(
    y_true,
    y_pred
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("CNN Baseline Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig(
    "results/cnn_baseline/confusion_matrix.png"
)

plt.close()

print("\nConfusion matrix saved.")
print("Classification report saved.")