"""
Fill the missing Real-World v2 cells of Table 19.

Same 600-photo set and the same preprocessing as evaluate_real_world.py.
Single-pass for the trained models. EXP-010 uses the original 3-model
ensemble with 5-view TTA. EXP-006 is the TACO model after rembg.

Also re-scores EXP-003, EXP-004 and EXP-008 as a check against the
published v2 figures (78.33%, 82.67%, 88.83%).

Run from the project root:
    ./venv/bin/python -m src.evaluation.evaluate_v2_table19
"""

import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.models import load_model

DATASET_PATH = "data/real_test_dataset"
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
OUTPUT_DIR = "results/v2_table19"
FOLDER_PREFIXES = ["", "✅"]

SINGLE_PASS = [
    ("EXP-001", "results/cnn_baseline/cnn_baseline.keras", lambda x: x / 255.0),
    ("EXP-002", "results/mobilenet/mobilenet.keras", mobilenet_preprocess),
    ("EXP-003", "results/efficientnet/efficientnet.keras", efficientnet_preprocess),
    ("EXP-004", "results/efficientnet_augmented/efficientnet_augmented.keras", efficientnet_preprocess),
    ("EXP-005", "results/efficientnet_taco/efficientnet_taco.keras", efficientnet_preprocess),
    ("EXP-007", "results/efficientnet_realwaste/efficientnet_realwaste.keras", efficientnet_preprocess),
    ("EXP-008", "results/efficientnet_household/efficientnet_household.keras", efficientnet_preprocess),
]

ENSEMBLE_PATHS = [
    "results/efficientnet_household/efficientnet_household.keras",
    "results/efficientnet_taco/efficientnet_taco.keras",
    "results/efficientnet_realwaste/efficientnet_realwaste.keras",
]

PUBLISHED = {
    "EXP-003": 470,
    "EXP-004": 496,
    "EXP-008": 533,
}


def resolve_class_dir(cls):
    for prefix in FOLDER_PREFIXES:
        candidate = os.path.join(DATASET_PATH, f"{prefix}{cls}")
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(cls)


def load_index():
    paths, labels = [], []
    for label_idx, cls in enumerate(CLASS_NAMES):
        cls_dir = resolve_class_dir(cls)
        names = sorted(
            f for f in os.listdir(cls_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        )
        for fname in names:
            paths.append(os.path.join(cls_dir, fname))
            labels.append(label_idx)
        print(f"  {cls}: {len(names)}")
    return paths, np.array(labels, dtype=np.int32)


def load_resized(paths):
    batch = np.empty((len(paths), *IMAGE_SIZE, 3), dtype=np.float32)
    for i, path in enumerate(paths):
        img = Image.open(path).convert("RGB").resize(IMAGE_SIZE)
        batch[i] = np.array(img, dtype=np.float32)
    return batch


def accuracy(y_true, y_pred):
    correct = int(np.sum(y_true == y_pred))
    return correct, correct / len(y_true)


def eval_single_pass(name, path, preprocess, x_raw, y_true):
    print(f"\n=== {name} ===")
    model = load_model(path, compile=False)
    probs = model.predict(preprocess(x_raw.copy()), batch_size=32, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    correct, acc = accuracy(y_true, y_pred)
    print(f"{name}: {correct}/{len(y_true)} = {acc * 100:.2f}%")
    del model
    return {"correct": correct, "n": int(len(y_true)), "accuracy": round(acc * 100, 2)}


def _center_crop(img, ratio):
    w, h = img.size
    nw, nh = int(w * ratio), int(h * ratio)
    left = (w - nw) // 2
    top = (h - nh) // 2
    return img.crop((left, top, left + nw, top + nh))


def tta_batch(paths):
    views = np.empty((len(paths), 5, *IMAGE_SIZE, 3), dtype=np.float32)
    for i, path in enumerate(paths):
        base = Image.open(path).convert("RGB")
        crops = [
            base,
            ImageOps.mirror(base),
            _center_crop(base, 0.85),
            _center_crop(base, 0.70),
            ImageOps.mirror(_center_crop(base, 0.85)),
        ]
        for v, crop in enumerate(crops):
            views[i, v] = np.array(crop.resize(IMAGE_SIZE), dtype=np.float32)
        if (i + 1) % 100 == 0:
            print(f"  views {i + 1}/{len(paths)}")
    return views


def eval_ensemble(paths, y_true):
    print("\n=== EXP-010 ensemble + TTA ===")
    views = tta_batch(paths)
    n, n_views = views.shape[:2]
    flat = efficientnet_preprocess(views.reshape(-1, *IMAGE_SIZE, 3).copy())
    acc_sum = None
    for path in ENSEMBLE_PATHS:
        print(f"  model {path}")
        model = load_model(path, compile=False)
        probs = model.predict(flat, batch_size=32, verbose=0).reshape(n, n_views, -1)
        mean = probs.mean(axis=1)
        acc_sum = mean if acc_sum is None else acc_sum + mean
        del model
    y_pred = np.argmax(acc_sum / len(ENSEMBLE_PATHS), axis=1)
    correct, acc = accuracy(y_true, y_pred)
    print(f"EXP-010: {correct}/{len(y_true)} = {acc * 100:.2f}%")
    return {"correct": correct, "n": int(len(y_true)), "accuracy": round(acc * 100, 2)}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Indexing real_test_dataset")
    paths, y_true = load_index()
    print(f"Total {len(paths)}")
    if len(paths) != 600:
        raise SystemExit(f"Expected 600 images, found {len(paths)}")

    print("Resizing for single-pass")
    x_raw = load_resized(paths)

    results = {}
    for name, path, preprocess in SINGLE_PASS:
        results[name] = eval_single_pass(name, path, preprocess, x_raw, y_true)
        if name in PUBLISHED and results[name]["correct"] != PUBLISHED[name]:
            print(
                f"  CHECK differs from published {PUBLISHED[name]}/600 "
                f"(got {results[name]['correct']}/600)"
            )

    out = os.path.join(OUTPUT_DIR, "single_pass.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out}")

    results["EXP-010"] = eval_ensemble(paths, y_true)
    out_all = os.path.join(OUTPUT_DIR, "results.json")
    with open(out_all, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_all}")
    for name, row in results.items():
        print(f"{name}\t{row['correct']}/{row['n']}\t{row['accuracy']:.2f}%")


if __name__ == "__main__":
    main()
