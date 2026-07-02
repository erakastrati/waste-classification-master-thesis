"""
prepare_garbage_v2.py  —  Map Garbage Classification v2 (10 classes) → 6 TrashNet classes.

Source: data/garbage-v2/  (unzipped Kaggle download)
  Expected: .../garbage-dataset/<class>/  or direct class folders under garbage-v2/

Output: data/garbage-v2-prepared/

Dataset: Garbage Classification v2 (Kaggle: sumn2u/garbage-classification-v2)
10 classes → 6 TrashNet classes.

Run from project root:
    python -m src.data.prepare_garbage_v2
"""

import os
import shutil

SOURCE_ROOT = "data/garbage-v2"
OUTPUT_ROOT = "data/garbage-v2-prepared"

# Garbage v2 folder names (lowercase) → TrashNet class
CLASS_MAP = {
    "cardboard":  "cardboard",
    "glass":      "glass",
    "metal":      "metal",
    "paper":      "paper",
    "plastic":    "plastic",
    "trash":      "trash",
    "biological": "trash",
    "battery":    "trash",
    "shoes":      "trash",
    "clothes":    "trash",
}

TRASHNET_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def find_class_root():
    """Locate folder containing class subdirectories."""
    candidates = [SOURCE_ROOT]
    for root, dirs, files in os.walk(SOURCE_ROOT):
        dir_names = {d.lower() for d in dirs}
        if "cardboard" in dir_names and "glass" in dir_names and "plastic" in dir_names:
            candidates.append(root)
    # Prefer deepest match (usually garbage-dataset/)
    candidates = [c for c in candidates if os.path.isdir(c)]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.count(os.sep))


def main():
    class_root = find_class_root()
    if class_root is None:
        raise FileNotFoundError(
            f"No class folders found under {SOURCE_ROOT}/.\n"
            "Download from Kaggle (sumn2u/garbage-classification-v2) and unzip to:\n"
            "  data/garbage-v2/garbage-dataset/"
        )

    print(f"Found class root: {class_root}")
    print(f"Output: {OUTPUT_ROOT}\n")

    counters = {cls: 0 for cls in TRASHNET_CLASSES}
    for cls in TRASHNET_CLASSES:
        os.makedirs(os.path.join(OUTPUT_ROOT, cls), exist_ok=True)

    for entry in sorted(os.listdir(class_root)):
        src_key = entry.lower()
        if src_key not in CLASS_MAP:
            continue

        dst_class = CLASS_MAP[src_key]
        src_dir = os.path.join(class_root, entry)
        if not os.path.isdir(src_dir):
            continue

        copied = 0
        for fname in sorted(os.listdir(src_dir)):
            if not fname.lower().endswith(IMAGE_EXTS):
                continue
            src_path = os.path.join(src_dir, fname)
            count = counters[dst_class]
            ext = os.path.splitext(fname)[1].lower()
            dst_name = f"garbage_{src_key}_{count:04d}{ext}"
            dst_path = os.path.join(OUTPUT_ROOT, dst_class, dst_name)
            shutil.copy2(src_path, dst_path)
            counters[dst_class] += 1
            copied += 1

        print(f"  {entry:15s} → {dst_class:10s}  {copied:4d} images")

    print("\n" + "=" * 40)
    print("GARBAGE V2 PREPARATION COMPLETE")
    print("=" * 40)
    total = 0
    for cls in TRASHNET_CLASSES:
        print(f"  {cls:12s}  {counters[cls]:4d} images")
        total += counters[cls]
    print(f"  {'TOTAL':12s}  {total:4d} images")


if __name__ == "__main__":
    main()
