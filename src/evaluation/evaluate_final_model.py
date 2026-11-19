"""
evaluate_final_model.py  —  Final evaluation: EfficientNetB0 + Garbage v2 (EXP-011) + TTA

Runs the saved model (efficientnet_garbage_best.keras) with the same 5-view TTA
used by app.py, over data/real_test_dataset (the current held-out set).

Results are written only to results/final_held_out_evaluation/.
The historical results/final_model_eval/ directory is not touched.

Run from project root:
    ./venv/bin/python -m src.evaluation.evaluate_final_model
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image, ImageOps
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from tensorflow.keras.models import load_model

from src.inference.tta_predict import predict_with_tta

MODEL_PATH   = "results/efficientnet_garbage/efficientnet_garbage_best.keras"
DATASET_PATH = "data/real_test_dataset"
IMAGE_SIZE   = (224, 224)
CLASS_NAMES  = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
OUTPUT_DIR   = "results/final_held_out_evaluation"

# The real_test_dataset folders are prefixed with a checkmark emoji (✅cardboard, ...).
FOLDER_PREFIXES = ["", "✅"]

if os.path.exists(os.path.join(OUTPUT_DIR, "summary.txt")):
    raise SystemExit(
        f"STOP: {OUTPUT_DIR}/summary.txt already exists; refusing to overwrite"
    )
os.makedirs(OUTPUT_DIR, exist_ok=True)


def resolve_class_dir(dataset_path, cls):
    for prefix in FOLDER_PREFIXES:
        candidate = os.path.join(dataset_path, f"{prefix}{cls}")
        if os.path.isdir(candidate):
            return candidate
    return None


def load_dataset():
    images, labels, paths, exif_rows = [], [], [], []
    for label_idx, cls in enumerate(CLASS_NAMES):
        cls_dir = resolve_class_dir(DATASET_PATH, cls)
        if cls_dir is None:
            print(f"  WARNING: no folder found for class '{cls}' in {DATASET_PATH}")
            continue
        for fname in sorted(os.listdir(cls_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue
            path = os.path.join(cls_dir, fname)
            with Image.open(path) as raw:
                orientation = raw.getexif().get(274, 1)
                stored_size = raw.size
                upright = ImageOps.exif_transpose(raw).convert("RGB")
            images.append(upright)
            labels.append(label_idx)
            paths.append(path)
            exif_rows.append((path, orientation, stored_size, upright.size))
    return images, np.array(labels), paths, exif_rows


print("Loading final model (EXP-011: EfficientNetB0 + Garbage v2 FT)...")
model = load_model(MODEL_PATH)

print("Loading real-world test set (600 photos)...")
images, y_true, paths, exif_rows = load_dataset()
print(f"Images loaded: {len(images)}")
n_rotated = sum(1 for _, orientation, _, _ in exif_rows if orientation != 1)
n_size_changed = sum(1 for _, _, stored, upright in exif_rows if stored != upright)
print(f"EXIF orientation not 1: {n_rotated}; size changed by transpose: {n_size_changed}")
with open(f"{OUTPUT_DIR}/exif_orientation.txt", "w") as f:
    f.write("EXIF correction is applied in memory by ImageOps.exif_transpose before RGB conversion.\n")
    f.write("Original JPEG files are not modified.\n")
    f.write(f"images={len(exif_rows)} orientation_not_1={n_rotated} stored_size_changed={n_size_changed}\n")

print("Running TTA inference (5 views per image)...")
y_pred = []
confidences = []
infer_ms = []
for i, img in enumerate(images):
    if (i + 1) % 50 == 0:
        print(f"  [{i + 1}/{len(images)}]")
    t0 = time.perf_counter()
    probs = predict_with_tta(model, img, IMAGE_SIZE)
    infer_ms.append((time.perf_counter() - t0) * 1000)
    pred_idx = int(np.argmax(probs))
    y_pred.append(pred_idx)
    confidences.append(float(probs[pred_idx]))
infer_ms = np.array(infer_ms)

y_pred = np.array(y_pred)
confidences = np.array(confidences)

# ==========================================
# Summary
# ==========================================

acc = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
n_correct = int(np.sum(y_true == y_pred))
n_errors = len(y_true) - n_correct

summary_lines = [
    "NEW held-out dataset (not the historical real_test_dataset that produced 92.17%).",
    "Final model: EfficientNetB0 + Garbage v2 Fine-tune (EXP-011) + TTA (5 views)",
    f"Model file:  {MODEL_PATH}",
    f"Dataset:     {DATASET_PATH} (n={len(y_true)})",
    "EXIF orientation applied in memory with ImageOps.exif_transpose; JPEG files unchanged.",
    "",
    f"Accuracy:   {n_correct}/{len(y_true)} = {acc*100:.2f}%",
    f"Errors:     {n_errors}",
    f"Precision:  {macro_precision:.4f} (macro)",
    f"Recall:     {macro_recall:.4f} (macro)",
    f"Macro F1:   {macro_f1:.4f}",
]
summary_text = "\n".join(summary_lines)
print("\n" + "=" * 50)
print(summary_text)
print("=" * 50)

with open(f"{OUTPUT_DIR}/summary.txt", "w") as f:
    f.write(summary_text + "\n")

# Single-image latency, same protocol as benchmark_inference.py
# (10 warmup + 100 timed batch-1 predicts). Written only in this directory.
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
sample = np.array(images[0].resize(IMAGE_SIZE), dtype=np.float32)
batch = np.expand_dims(efficientnet_preprocess(sample.copy()), axis=0)
for _ in range(10):
    model.predict(batch, verbose=0)
single_ms = []
for _ in range(100):
    t0 = time.perf_counter()
    model.predict(batch, verbose=0)
    single_ms.append((time.perf_counter() - t0) * 1000)
single_ms = np.array(single_ms)
infer_lines = [
    "Inference timing for the NEW held-out evaluation.",
    "TTA times are the actual 5-view calls on all 600 photographs.",
    "Single-pass times follow benchmark_inference.py: 10 warmup + 100 runs, batch=1, one image.",
    f"sample_image={paths[0]}",
    "",
    f"TTA per image: mean={infer_ms.mean():.2f} ms  std={infer_ms.std():.2f}  median={np.median(infer_ms):.2f}  n={len(infer_ms)}",
    f"TTA total: {infer_ms.sum()/1000:.1f} s",
    f"Single-pass: mean={single_ms.mean():.2f} ms  std={single_ms.std():.2f}  median={np.median(single_ms):.2f}  n=100",
    f"Single-pass x5 (estimate of TTA): {single_ms.mean()*5:.2f} ms",
]
with open(f"{OUTPUT_DIR}/inference_time.txt", "w") as f:
    f.write("\n".join(infer_lines) + "\n")
print("\n".join(infer_lines))

# ==========================================
# Classification report (precision/recall/F1 per class)
# ==========================================

report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
with open(f"{OUTPUT_DIR}/classification_report.txt", "w") as f:
    f.write(summary_text + "\n\n" + report)
print("\n" + report)

# ==========================================
# Confusion matrix
# ==========================================

cm = confusion_matrix(y_true, y_pred)
per_prec = precision_score(y_true, y_pred, average=None, zero_division=0, labels=range(len(CLASS_NAMES)))
per_rec = recall_score(y_true, y_pred, average=None, zero_division=0, labels=range(len(CLASS_NAMES)))
per_f1 = f1_score(y_true, y_pred, average=None, zero_division=0, labels=range(len(CLASS_NAMES)))
with open(f"{OUTPUT_DIR}/per_class_metrics.csv", "w") as f:
    f.write("class,n,correct,accuracy,precision,recall,f1\n")
    for i, cls in enumerate(CLASS_NAMES):
        n_i = int(cm[i].sum())
        correct_i = int(cm[i, i])
        acc_i = correct_i / n_i if n_i else 0.0
        f.write(f"{cls},{n_i},{correct_i},{acc_i:.4f},{per_prec[i]:.4f},{per_rec[i]:.4f},{per_f1[i]:.4f}\n")
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
)
plt.title(f"Final Model (EXP-011 + TTA) — Real-World Confusion Matrix (n={len(y_true)}, acc={acc*100:.2f}%)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=150)
plt.close()

# ==========================================
# Full misclassified list (with confidence), same format as final_misclassified.txt
# ==========================================

pair_counts = {}
lines = [
    "Mode: TTA (5 views, matches app.py inference)",
    f"Accuracy: {n_correct}/{len(y_true)} = {acc*100:.2f}%",
    f"Errors: {n_errors}",
    "",
    "Pairs:",
]

error_rows = []
for true_idx, pred_idx, conf, path in zip(y_true, y_pred, confidences, paths):
    if true_idx != pred_idx:
        key = (CLASS_NAMES[true_idx], CLASS_NAMES[pred_idx])
        pair_counts[key] = pair_counts.get(key, 0) + 1
        error_rows.append((CLASS_NAMES[true_idx], CLASS_NAMES[pred_idx], conf, path))

for (true_c, pred_c), count in sorted(pair_counts.items(), key=lambda kv: -kv[1]):
    lines.append(f"  {true_c} → {pred_c}: {count}")

lines.append("")
lines.append("Files:")
for true_c, pred_c, conf, path in error_rows:
    lines.append(f"{true_c} → {pred_c} ({conf*100:.1f}%) | {path}")

with open(f"{OUTPUT_DIR}/misclassified.txt", "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"\nAll artifacts saved to {OUTPUT_DIR}/")
print("  - confusion_matrix.png")
print("  - classification_report.txt")
print("  - misclassified.txt")
print("  - summary.txt")

# ==========================================
# Confidence-stratified accuracy (all 600 predictions, correct + incorrect)
# ==========================================

BINS = [(0.0, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.001)]
bin_labels = ["<60%", "60-70%", "70-80%", "80-90%", ">=90%"]

conf_lines = [
    "Confidence-stratified accuracy (n=600, TTA confidence = max softmax after 5-view averaging)",
    "",
    f"{'Bucket':<10}{'n':>6}{'% of set':>10}{'correct':>9}{'accuracy':>10}",
]

bucket_rows = []
for (lo, hi), label in zip(BINS, bin_labels):
    mask = (confidences >= lo) & (confidences < hi)
    n = int(mask.sum())
    if n == 0:
        bucket_rows.append((label, 0, 0, 0.0))
        continue
    correct = int((y_true[mask] == y_pred[mask]).sum())
    bucket_acc = correct / n
    bucket_rows.append((label, n, correct, bucket_acc))

for label, n, correct, bucket_acc in bucket_rows:
    pct_of_set = n / len(y_true) * 100
    conf_lines.append(f"{label:<10}{n:>6}{pct_of_set:>9.1f}%{correct:>9}{bucket_acc*100:>9.1f}%")

with open(f"{OUTPUT_DIR}/confidence_stratified.txt", "w") as f:
    f.write("\n".join(conf_lines) + "\n")

print("\n" + "\n".join(conf_lines))
print("  - confidence_stratified.txt")

# ==========================================
# Breakdown aligned with the app's actual warning thresholds (<50%, <70%)
# See templates/index.html lines ~766-774
# ==========================================

APP_BINS = [(0.0, 0.50), (0.50, 0.70), (0.70, 1.001)]
app_bin_labels = ["<50% (red warning)", "50-70% (orange warning)", ">=70% (no warning)"]

app_lines = [
    "Breakdown aligned with production warning thresholds (templates/index.html)",
    "",
    f"{'Bucket':<26}{'n':>6}{'% of set':>10}{'correct':>9}{'accuracy':>10}",
]
for (lo, hi), label in zip(APP_BINS, app_bin_labels):
    mask = (confidences >= lo) & (confidences < hi)
    n = int(mask.sum())
    correct = int((y_true[mask] == y_pred[mask]).sum()) if n else 0
    acc_b = correct / n if n else 0.0
    app_lines.append(f"{label:<26}{n:>6}{n/len(y_true)*100:>9.1f}%{correct:>9}{acc_b*100:>9.1f}%")

with open(f"{OUTPUT_DIR}/confidence_vs_app_thresholds.txt", "w") as f:
    f.write("\n".join(app_lines) + "\n")

print("\n" + "\n".join(app_lines))
print("  - confidence_vs_app_thresholds.txt")

# ==========================================
# Raw per-image confidence dump (for any future re-binning without re-running TTA)
# ==========================================

with open(f"{OUTPUT_DIR}/all_predictions_confidence.csv", "w") as f:
    f.write("true_class,pred_class,confidence,correct,path\n")
    for t, p, c, path in zip(y_true, y_pred, confidences, paths):
        f.write(f"{CLASS_NAMES[t]},{CLASS_NAMES[p]},{c:.4f},{int(t==p)},{path}\n")

print("  - all_predictions_confidence.csv")
