"""
prepare_realwaste.py  —  Map RealWaste (7 folders) to 6 TrashNet classes.

RealWaste source: data/realwaste-dataset/
Output:           data/realwaste-prepared/

Mapping:
  Cardboard            → cardboard
  Glass                → glass
  Metal                → metal
  Paper                → paper
  Plastic              → plastic
  Miscellaneous Trash  → trash
  Textile Trash        → trash

Run from project root:
    python -m src.data.prepare_realwaste
"""

import os
import shutil

SOURCE_ROOT = "data/realwaste-dataset"
OUTPUT_ROOT = "data/realwaste-prepared"

CLASS_MAP = {
    "Cardboard":           "cardboard",
    "Glass":               "glass",
    "Metal":               "metal",
    "Paper":               "paper",
    "Plastic":             "plastic",
    "Miscellaneous Trash": "trash",
    "Textile Trash":       "trash",
}

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
counters = {cls: 0 for cls in classes}

for cls in classes:
    os.makedirs(os.path.join(OUTPUT_ROOT, cls), exist_ok=True)

print("Preparing RealWaste dataset...")
print(f"Source: {SOURCE_ROOT}")
print(f"Output: {OUTPUT_ROOT}\n")

for src_folder, dst_class in CLASS_MAP.items():
    src_dir = os.path.join(SOURCE_ROOT, src_folder)
    if not os.path.isdir(src_dir):
        print(f"  WARNING: missing folder {src_folder}")
        continue

    prefix = src_folder.lower().replace(" ", "_")
    copied = 0

    for fname in os.listdir(src_dir):
        if not fname.lower().endswith(IMAGE_EXTS):
            continue

        src_path = os.path.join(src_dir, fname)
        count    = counters[dst_class]
        dst_name = f"{prefix}_{count:04d}{os.path.splitext(fname)[1].lower()}"
        dst_path = os.path.join(OUTPUT_ROOT, dst_class, dst_name)

        shutil.copy2(src_path, dst_path)
        counters[dst_class] += 1
        copied += 1

    print(f"  {src_folder:22s} → {dst_class:10s}  {copied:4d} images")

print("\n" + "=" * 40)
print("REALWASTE PREPARATION COMPLETE")
print("=" * 40)
total = 0
for cls in classes:
    print(f"  {cls:12s}  {counters[cls]:4d} images")
    total += counters[cls]
print(f"  {'TOTAL':12s}  {total:4d} images")
