"""Render TrashNet pair contact sheets and the pending review register.

Reads existing audit tables. Writes only under results/retraining_2026/.
Does not modify source images, splits, checkpoints, or the earlier pair CSV.
"""

import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[5]
AUDIT = ROOT / "results/retraining_2026/audit/EXP-004"
PAIRS = AUDIT / "cross_split_pairs.csv"
SUPPLEMENTARY = ROOT / "results/duplicate_check/supplementary_trashnet_candidates.csv"
OUT = AUDIT / "visual_review"
REGISTER = AUDIT / "visual_review_register.csv"

FONT = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 18)
FONT_B = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 22)
FONT_S = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 15)

PAGE_W = 1680
MARGIN = 28
IMG_W = 720
IMG_H = 480
PAIRS_PER_SHEET = 2
GAP = 24

WITHIN_TRAIN = {
    "pair_id": "P139",
    "component_id": "WITHIN_TRAIN",
    "component_size": "2",
    "image_a": "data/trashNet-dataset/glass/glass115.jpg",
    "class_a": "glass",
    "split_a": "train",
    "image_b": "data/trashNet-dataset/metal/metal91.jpg",
    "class_b": "metal",
    "split_b": "train",
    "same_class": "no",
    "phash_dist": "0",
    "dhash_dist": "0",
    "min_hash_dist": "0",
    "zncc": "1.0000",
    "automatic_verdict": "confirmed",
    "byte_identical": "yes",
    "review_priority": "1_exact_bytes_within_training",
    "evidence_note": (
        "SHA-256 match, both files in the shuffled training list, "
        "opposite folder labels. Not one of the 138 cross-split pairs."
    ),
}


def membership(split):
    if split.startswith("train"):
        base = "training"
    elif split.startswith("val"):
        base = "validation"
    else:
        base = split
    if "ftval" in split:
        return base + "; also unshuffled fine-tune validation"
    return base


def band_of(row):
    priority = row["review_priority"]
    if priority == "1_exact_bytes":
        return "01_exact_cross_split"
    if priority == "1_exact_bytes_within_training":
        return "02_exact_within_training"
    if priority == "2_near_automatic":
        return "03_high_similarity_paper"
    if priority == "3_opposite_class_automatic":
        return "04_opposite_class"
    return "05_by_component/" + row["component_id"]


def header_color(band):
    if band.startswith("01") or band.startswith("02"):
        return (120, 32, 32)
    if band.startswith("03"):
        return (140, 90, 20)
    if band.startswith("04"):
        return (30, 70, 120)
    return (50, 50, 50)


def load_pairs():
    with PAIRS.open() as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 138:
        raise SystemExit(f"Expected 138 cross-split pairs, found {len(rows)}")
    within = dict(WITHIN_TRAIN)
    with SUPPLEMENTARY.open() as handle:
        for record in csv.DictReader(handle):
            paths = {record["trashnet_image"], record["matched_image"]}
            if paths == {within["image_a"], within["image_b"]} and record["compared_with"] == "trashnet":
                within["phash_dist"] = record["phash_dist"]
                within["dhash_dist"] = record["dhash_dist"]
                within["zncc"] = record["zncc"]
                within["min_hash_dist"] = str(min(int(record["phash_dist"]), int(record["dhash_dist"])))
                within["automatic_verdict"] = record["verdict"]
                within["split_a"] = record["trashnet_split"] if record["trashnet_image"].endswith("glass115.jpg") else record["matched_split"]
                within["split_b"] = record["matched_split"] if record["matched_image"].endswith("metal91.jpg") else record["trashnet_split"]
                break
        else:
            raise SystemExit("Within-training exact pair was not found in the supplementary CSV")
    return rows, within


def fit(path, box):
    source = ROOT / path
    if not source.is_file():
        tile = Image.new("RGB", box, (230, 230, 230))
        draw = ImageDraw.Draw(tile)
        draw.text((16, box[1] // 2), "MISSING FILE", fill=(160, 0, 0), font=FONT_B)
        return tile, False
    with Image.open(source) as image:
        image = image.convert("RGB")
        image.thumbnail(box, Image.Resampling.LANCZOS)
        tile = Image.new("RGB", box, "white")
        x = (box[0] - image.width) // 2
        y = (box[1] - image.height) // 2
        tile.paste(image, (x, y))
    return tile, True


def draw_pair(canvas, top, row, band):
    draw = ImageDraw.Draw(canvas)
    left = MARGIN
    right = PAGE_W - MARGIN
    head = 78
    draw.rectangle((left, top, right, top + head), fill=header_color(band))
    title = (
        f"{row['pair_id']}    {band.split('/')[-1]}    "
        f"component {row['component_id']} (size {row['component_size']})"
    )
    draw.text((left + 12, top + 8), title, fill="white", font=FONT_B)
    metrics = (
        f"pHash {row['phash_dist']}    dHash {row['dhash_dist']}    "
        f"min hash {row['min_hash_dist']}    ZNCC {row['zncc']}    "
        f"byte-identical {row['byte_identical']}    visual review PENDING"
    )
    draw.text((left + 12, top + 40), metrics, fill="white", font=FONT_S)
    image_top = top + head + 12
    positions = (left, left + IMG_W + GAP)
    sides = (
        (row["image_a"], row["class_a"], row["split_a"]),
        (row["image_b"], row["class_b"], row["split_b"]),
    )
    missing = []
    for origin, (path, label, split) in zip(positions, sides):
        tile, present = fit(path, (IMG_W, IMG_H))
        if not present:
            missing.append(path)
        canvas.paste(tile, (origin, image_top))
        draw.rectangle((origin, image_top, origin + IMG_W - 1, image_top + IMG_H - 1), outline=(180, 180, 180))
        caption_y = image_top + IMG_H + 8
        draw.text((origin, caption_y), Path(path).name, fill="black", font=FONT_B)
        draw.text((origin, caption_y + 28), f"class {label}    {membership(split)}", fill="black", font=FONT)
        draw.text((origin, caption_y + 52), path, fill=(60, 60, 60), font=FONT_S)
    return missing


def block_height():
    return 78 + 12 + IMG_H + 8 + 76 + 16


def write_sheets(groups):
    index_rows = []
    missing = []
    sheet_count = 0
    for band, rows in groups:
        directory = OUT / band
        directory.mkdir(parents=True, exist_ok=True)
        for start in range(0, len(rows), PAIRS_PER_SHEET):
            chunk = rows[start:start + PAIRS_PER_SHEET]
            height = MARGIN + block_height() * len(chunk) + MARGIN
            canvas = Image.new("RGB", (PAGE_W, height), "white")
            top = MARGIN
            for row in chunk:
                missing.extend(draw_pair(canvas, top, row, band))
                top += block_height()
            sheet_count += 1
            name = f"sheet_{start // PAIRS_PER_SHEET + 1:02d}.png"
            destination = directory / name
            canvas.save(destination, format="PNG")
            relative = destination.relative_to(ROOT).as_posix()
            for row in chunk:
                index_rows.append({
                    "pair_id": row["pair_id"],
                    "band": band,
                    "sheet": relative,
                })
    return sheet_count, index_rows, missing


def main():
    rows, within = load_pairs()
    bands = defaultdict(list)
    for row in rows:
        bands[band_of(row)].append(row)
    bands[band_of(within)].append(within)
    order = ["01_exact_cross_split", "02_exact_within_training", "03_high_similarity_paper", "04_opposite_class"]
    component_bands = sorted(key for key in bands if key.startswith("05_"))
    grouped = [(name, bands[name]) for name in order + component_bands]
    covered = sum(len(items) for _, items in grouped)
    if covered != 139:
        raise SystemExit(f"Expected 139 pairs on sheets, found {covered}")
    sheet_count, index_rows, missing = write_sheets(grouped)
    by_id = {row["pair_id"]: row for row in index_rows}
    register_fields = [
        "pair_id", "component_id", "component_size",
        "image_a", "class_a", "split_a", "membership_a",
        "image_b", "class_b", "split_b", "membership_b",
        "same_class", "phash_dist", "dhash_dist", "min_hash_dist", "zncc",
        "exact_duplicate", "automatic_verdict", "review_band", "sheet",
        "visual_review_status", "reviewer_decision", "decision_rationale",
        "proposed_action", "action_requirement",
    ]
    all_rows = rows + [within]
    with REGISTER.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=register_fields)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({
                "pair_id": row["pair_id"],
                "component_id": row["component_id"],
                "component_size": row["component_size"],
                "image_a": row["image_a"],
                "class_a": row["class_a"],
                "split_a": row["split_a"],
                "membership_a": membership(row["split_a"]),
                "image_b": row["image_b"],
                "class_b": row["class_b"],
                "split_b": row["split_b"],
                "membership_b": membership(row["split_b"]),
                "same_class": row["same_class"],
                "phash_dist": row["phash_dist"],
                "dhash_dist": row["dhash_dist"],
                "min_hash_dist": row["min_hash_dist"],
                "zncc": row["zncc"],
                "exact_duplicate": row["byte_identical"],
                "automatic_verdict": row["automatic_verdict"],
                "review_band": band_of(row),
                "sheet": by_id[row["pair_id"]]["sheet"],
                "visual_review_status": "PENDING",
                "reviewer_decision": "",
                "decision_rationale": "",
                "proposed_action": "PENDING",
                "action_requirement": "PENDING",
            })
    with (OUT / "sheet_index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pair_id", "band", "sheet"])
        writer.writeheader()
        writer.writerows(index_rows)
    print(f"sheets {sheet_count}")
    print(f"pairs {covered}")
    print(f"missing {len(missing)}")
    for path in missing:
        print("MISSING", path)
    for name, items in grouped:
        print(f"{name} pairs {len(items)} sheets {(len(items) + PAIRS_PER_SHEET - 1) // PAIRS_PER_SHEET}")


if __name__ == "__main__":
    main()
