"""
prepare_household.py  —  Map Household Waste (real_world only) to 6 TrashNet classes.

Source: data/household-waste/images/<category>/real_world/
Output: data/local_train/

Dataset: Recyclable and Household Waste Classification (Kaggle: alistairking)
License: check Kaggle dataset page for thesis citation.

Run from project root:
    python -m src.data.prepare_household
"""

import os
import shutil

SOURCE_ROOT = "data/household-waste/images"
OUTPUT_ROOT = "data/local_train"
SUBFOLDER   = "real_world"

CLASS_MAP = {
    # cardboard
    "cardboard_boxes":     "cardboard",
    "cardboard_packaging": "cardboard",
    # glass
    "glass_beverage_bottles":    "glass",
    "glass_cosmetic_containers": "glass",
    "glass_food_jars":           "glass",
    # metal
    "aerosol_cans":        "metal",
    "aluminum_food_cans":  "metal",
    "aluminum_soda_cans":  "metal",
    "steel_food_cans":     "metal",
    # paper
    "magazines":    "paper",
    "newspaper":    "paper",
    "office_paper": "paper",
    # plastic
    "disposable_plastic_cutlery": "plastic",
    "plastic_cup_lids":           "plastic",
    "plastic_detergent_bottles":  "plastic",
    "plastic_food_containers":    "plastic",
    "plastic_shopping_bags":      "plastic",
    "plastic_soda_bottles":       "plastic",
    "plastic_straws":             "plastic",
    "plastic_trash_bags":         "plastic",
    "plastic_water_bottles":      "plastic",
    "styrofoam_cups":             "plastic",
    "styrofoam_food_containers":  "plastic",
    "paper_cups":                 "plastic",
    # trash
    "clothing":       "trash",
    "coffee_grounds": "trash",
    "eggshells":      "trash",
    "food_waste":     "trash",
    "shoes":          "trash",
    "tea_bags":       "trash",
}

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
CLASSES    = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
counters   = {cls: 0 for cls in CLASSES}

print(f"  {'TOTAL':12s}  {total:4d} images")


def main():
    global counters
    counters = {cls: 0 for cls in CLASSES}
    for cls in CLASSES:
        os.makedirs(os.path.join(OUTPUT_ROOT, cls), exist_ok=True)

    print("Preparing Household Waste dataset (real_world only)...")
    print(f"Source: {SOURCE_ROOT}/<category>/{SUBFOLDER}/")
    print(f"Output: {OUTPUT_ROOT}/\n")

    for src_cat, dst_class in CLASS_MAP.items():
        src_dir = os.path.join(SOURCE_ROOT, src_cat, SUBFOLDER)
        if not os.path.isdir(src_dir):
            print(f"  WARNING: missing {src_cat}/{SUBFOLDER}")
            continue

        copied = 0
        for fname in sorted(os.listdir(src_dir)):
            if not fname.lower().endswith(IMAGE_EXTS):
                continue

            src_path = os.path.join(src_dir, fname)
            count    = counters[dst_class]
            ext      = os.path.splitext(fname)[1].lower()
            dst_name = f"household_{src_cat}_{count:04d}{ext}"
            dst_path = os.path.join(OUTPUT_ROOT, dst_class, dst_name)

            shutil.copy2(src_path, dst_path)
            counters[dst_class] += 1
            copied += 1

        print(f"  {src_cat:35s} → {dst_class:10s}  {copied:4d} images")

    print("\n" + "=" * 40)
    print("HOUSEHOLD PREPARATION COMPLETE")
    print("=" * 40)
    total = 0
    for cls in CLASSES:
        print(f"  {cls:12s}  {counters[cls]:4d} images")
        total += counters[cls]
    print(f"  {'TOTAL':12s}  {total:4d} images")


if __name__ == "__main__":
    main()
