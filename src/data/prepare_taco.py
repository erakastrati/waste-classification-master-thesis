"""
prepare_taco.py

Reads TACO COCO annotations, maps the 60 TACO categories to the 6
TrashNet classes, crops each annotated object using its bounding box,
and saves the crops to:

    data/taco-prepared/
        cardboard/
        glass/
        metal/
        paper/
        plastic/
        trash/

Run from project root:
    python -m src.data.prepare_taco
"""

import os
import json
from PIL import Image

# ==========================================
# Paths
# ==========================================

ANNOTATIONS_PATH = "data/taco-dataset/data/annotations.json"
IMAGES_ROOT      = "data/taco-dataset/data"
OUTPUT_ROOT      = "data/taco-prepared"

# Minimum bounding box size (pixels) — skip tiny objects
MIN_SIZE = 48

# ==========================================
# Category Mapping: TACO name → TrashNet class
# ==========================================

CATEGORY_MAP = {
    # --- cardboard ---
    "Corrugated carton":       "cardboard",
    "Other carton":            "cardboard",
    "Meal carton":             "cardboard",
    "Drink carton":            "cardboard",
    "Egg carton":              "cardboard",
    "Pizza box":               "cardboard",
    "Toilet tube":             "cardboard",
    "Paper bag":               "cardboard",
    "Plastified paper bag":    "cardboard",

    # --- glass ---
    "Glass bottle":            "glass",
    "Broken glass":            "glass",
    "Glass cup":               "glass",
    "Glass jar":               "glass",

    # --- metal ---
    "Drink can":               "metal",
    "Food Can":                "metal",
    "Aerosol":                 "metal",
    "Metal bottle cap":        "metal",
    "Pop tab":                 "metal",
    "Metal lid":               "metal",
    "Scrap metal":             "metal",
    "Aluminium foil":          "metal",
    "Aluminium blister pack":  "metal",

    # --- paper ---
    "Normal paper":            "paper",
    "Magazine paper":          "paper",
    "Tissues":                 "paper",
    "Wrapping paper":          "paper",
    "Paper cup":               "paper",
    "Paper straw":             "paper",

    # --- plastic ---
    "Clear plastic bottle":    "plastic",
    "Other plastic bottle":    "plastic",
    "Plastic film":            "plastic",
    "Plastic bottle cap":      "plastic",
    "Plastic lid":             "plastic",
    "Disposable plastic cup":  "plastic",
    "Foam cup":                "plastic",
    "Other plastic cup":       "plastic",
    "Styrofoam piece":         "plastic",
    "Plastic straw":           "plastic",
    "Single-use carrier bag":  "plastic",
    "Polypropylene bag":       "plastic",
    "Garbage bag":             "plastic",
    "Crisp packet":            "plastic",
    "Other plastic wrapper":   "plastic",
    "Plastic glooves":         "plastic",
    "Other plastic":           "plastic",
    "Six pack rings":          "plastic",
    "Spread tub":              "plastic",
    "Tupperware":              "plastic",
    "Disposable food container": "plastic",
    "Foam food container":     "plastic",
    "Other plastic container": "plastic",
    "Plastic utensils":        "plastic",
    "Squeezable tube":         "plastic",
    "Blister pack":            "plastic",
    "Carded blister pack":     "plastic",

    # --- trash ---
    "Cigarette":               "trash",
    "Unlabeled litter":        "trash",
    "Rope & strings":          "trash",
    "Shoe":                    "trash",
    "Food waste":              "trash",
    "Battery":                 "trash",
}

# ==========================================
# Create output directories
# ==========================================

classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
for cls in classes:
    os.makedirs(os.path.join(OUTPUT_ROOT, cls), exist_ok=True)


# ==========================================
# Load annotations
# ==========================================

print("Loading TACO annotations...")
with open(ANNOTATIONS_PATH) as f:
    data = json.load(f)

# Build lookup maps
cat_id_to_name = {c["id"]: c["name"] for c in data["categories"]}
img_id_to_info = {img["id"]: img  for img in data["images"]}

print(f"Total images:      {len(data['images'])}")
print(f"Total annotations: {len(data['annotations'])}")
print()

# ==========================================
# Group annotations by image_id (open each image only once)
# ==========================================

from collections import defaultdict

anns_by_image = defaultdict(list)
for ann in data["annotations"]:
    anns_by_image[ann["image_id"]].append(ann)

# ==========================================
# Crop and save
# ==========================================

counters = {cls: 0 for cls in classes}
skipped  = 0
unmapped = set()

total_images = len(anns_by_image)
print(f"Processing {total_images} images...\n")

for i, (img_id, annotations) in enumerate(anns_by_image.items()):
    if i % 100 == 0:
        print(f"  [{i:4d}/{total_images}] processed so far: {sum(counters.values())} crops saved")

    img_info = img_id_to_info.get(img_id)
    if img_info is None:
        skipped += len(annotations)
        continue

    img_path = os.path.join(IMAGES_ROOT, img_info["file_name"])
    if not os.path.exists(img_path):
        skipped += len(annotations)
        continue

    try:
        img    = Image.open(img_path).convert("RGB")
        iw, ih = img.size
    except Exception:
        skipped += len(annotations)
        continue

    for ann in annotations:
        cat_name     = cat_id_to_name.get(ann["category_id"], "")
        trashnet_cls = CATEGORY_MAP.get(cat_name)

        if trashnet_cls is None:
            unmapped.add(cat_name)
            skipped += 1
            continue

        x, y, w, h = ann["bbox"]
        x, y, w, h = int(x), int(y), int(w), int(h)

        if w < MIN_SIZE or h < MIN_SIZE:
            skipped += 1
            continue

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(iw, x + w)
        y2 = min(ih, y + h)

        try:
            crop     = img.crop((x1, y1, x2, y2))
            count    = counters[trashnet_cls]
            out_name = f"{trashnet_cls}_{count:04d}.jpg"
            out_path = os.path.join(OUTPUT_ROOT, trashnet_cls, out_name)
            crop.save(out_path, "JPEG", quality=90)
            counters[trashnet_cls] += 1
        except Exception:
            skipped += 1
            continue

# ==========================================
# Summary
# ==========================================

print("=" * 40)
print("TACO PREPARATION COMPLETE")
print("=" * 40)
print()
print("Crops saved per class:")
total = 0
for cls in classes:
    print(f"  {cls:12s}  {counters[cls]:4d} images")
    total += counters[cls]
print(f"  {'TOTAL':12s}  {total:4d} images")
print()
print(f"Skipped: {skipped}  (too small, unmapped, or missing file)")

if unmapped:
    print()
    print("Unmapped TACO categories (not assigned to any class):")
    for name in sorted(unmapped):
        print(f"  - {name}")

print()
print(f"Output: {OUTPUT_ROOT}/")
