"""Build the corrected TrashNet train/validation manifests.

Reads the human-review register, records the completed review for pair IDs
that were still unresolved, and writes one stratified group-aware split.
Does not move, relabel, or delete dataset files, and does not train.
"""

import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REGISTER = ROOT / "results/retraining_2026/audit/EXP-004/visual_review_register.csv"
OUT = ROOT / "results/retraining_2026/manifests/trashnet_seed42"
DATA = ROOT / "data/trashNet-dataset"
CLASSES = ("cardboard", "glass", "metal", "paper", "plastic", "trash")
SEED = 42

PRESERVE = {
    "P001": "keep_one_exclude_other",
    "P002": "keep_one_exclude_other",
    "P139": "keep_one_exclude_other",
    "P003": "visual_duplicate",
    "P004": "visual_duplicate",
    "P005": "visual_duplicate",
    "P006": "visual_duplicate",
    "P007": "visual_duplicate",
    "P037": "visual_duplicate",
    "P039": "visual_duplicate",
    "P038": "same_object_different_photo",
    "P010": "unrelated",
    "P046": "unrelated",
    "P053": "unrelated",
    "P126": "unrelated",
    "P132": "unrelated",
    "P133": "unrelated",
}
RECHECKED = {"P008", "P009", "P106"}
GROUPING = {"visual_duplicate", "same_object_different_photo"}
EXCLUSIONS = {
    "data/trashNet-dataset/glass/glass176.jpg",
    "data/trashNet-dataset/plastic/plastic332.jpg",
    "data/trashNet-dataset/metal/metal91.jpg",
}
KEPT = {
    "data/trashNet-dataset/plastic/plastic152.jpg": "plastic",
    "data/trashNet-dataset/glass/glass389.jpg": "glass",
    "data/trashNet-dataset/glass/glass115.jpg": "glass",
}


def pair_number(pair_id):
    return int(pair_id[1:])


def remaining_scope():
    numbers = list(range(8, 37)) + list(range(40, 139))
    return {f"P{n:03d}" for n in numbers}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def load_register():
    with REGISTER.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames)
        rows = list(reader)
    return fields, rows


def write_register(fields, rows):
    with REGISTER.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def apply_remaining_decisions(rows):
    expected = remaining_scope() - set(PRESERVE)
    unresolved = {row["pair_id"] for row in rows if row["visual_review_status"] == "UNRESOLVED"}
    if unresolved != expected:
        missing = sorted(expected - unresolved, key=pair_number)
        extra = sorted(unresolved - expected, key=pair_number)
        raise SystemExit(
            "Unresolved IDs do not match the completed-review statement. "
            f"missing_from_unresolved={missing} unexpected_unresolved={extra}"
        )
    for row in rows:
        if row["pair_id"] not in expected:
            continue
        if row["pair_id"] in RECHECKED:
            rationale = (
                "Explicitly rechecked during the completed review of all 139 pairs. "
                "Confirmed as different images with correct labels. The two files are not grouped."
            )
        else:
            rationale = (
                "Manual review of all 139 pairs. This pair has no problematic visual duplication "
                "and no incorrect label. The two files are not grouped."
            )
        row["visual_review_status"] = "HUMAN_CONFIRMED"
        row["reviewer_decision"] = "unrelated"
        row["decision_rationale"] = rationale
        row["proposed_action"] = "no_action"
        row["action_requirement"] = "no_action"
        row["decision_source"] = "human_review_2026-11-18"
        row["group_id"] = ""
        row["kept_path"] = ""
        row["excluded_path"] = ""


def class_parent(rel):
    return rel.split("/")[2]


def find_parent(items):
    parent = {item: item for item in items}

    def find(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left, right):
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    return find, union


def validate(rows):
    errors = []
    ids = [row["pair_id"] for row in rows]
    expected_ids = [f"P{n:03d}" for n in range(1, 140)]
    if ids != expected_ids and sorted(ids, key=pair_number) != expected_ids:
        errors.append(f"pair IDs are not P001–P139: got {len(ids)} rows")
    if len(set(ids)) != 139:
        errors.append("pair IDs are not unique")

    for row in rows:
        if row["visual_review_status"] != "HUMAN_CONFIRMED":
            errors.append(f"{row['pair_id']} status is {row['visual_review_status']}")
        if row["pair_id"] in PRESERVE and row["reviewer_decision"] != PRESERVE[row["pair_id"]]:
            errors.append(
                f"{row['pair_id']} decision {row['reviewer_decision']} "
                f"does not preserve {PRESERVE[row['pair_id']]}"
            )
        if row["pair_id"] in remaining_scope() and row["reviewer_decision"] != "unrelated":
            errors.append(f"{row['pair_id']} was not recorded as unrelated")

    grouping_rows = [row for row in rows if row["reviewer_decision"] in GROUPING]
    paths = [row["image_a"] for row in grouping_rows] + [row["image_b"] for row in grouping_rows]
    find, union = find_parent(paths)
    for row in grouping_rows:
        if row["class_a"] != row["class_b"]:
            errors.append(f"{row['pair_id']} groups two classes")
        if not row["group_id"]:
            errors.append(f"{row['pair_id']} has no group id")
        union(row["image_a"], row["image_b"])

    components = defaultdict(list)
    for path in paths:
        components[find(path)].append(path)
    for members in components.values():
        classes = {class_parent(path) for path in members}
        if len(classes) != 1:
            errors.append(f"group mixes classes: {members}")
        if any(path in EXCLUSIONS for path in members):
            errors.append(f"excluded file is inside a keep-together group: {members}")

    grouped = set(paths)
    for row in rows:
        if row["reviewer_decision"] != "unrelated":
            continue
        pair_paths = {row["image_a"], row["image_b"]}
        if pair_paths <= grouped and find(row["image_a"]) == find(row["image_b"]):
            errors.append(f"{row['pair_id']} is unrelated but both files are in one group")

    excluded_seen = {row["excluded_path"] for row in rows if row["excluded_path"]}
    kept_seen = {row["kept_path"] for row in rows if row["kept_path"]}
    if excluded_seen != EXCLUSIONS:
        errors.append(f"exclusion paths {sorted(excluded_seen)} != {sorted(EXCLUSIONS)}")
    if kept_seen != set(KEPT):
        errors.append(f"kept paths {sorted(kept_seen)} != {sorted(KEPT)}")
    for path, label in KEPT.items():
        if class_parent(path) != label:
            errors.append(f"kept file {path} is not in folder {label}")
    if grouped & EXCLUSIONS:
        errors.append("a grouped file is also excluded")
    if grouped & set(KEPT):
        errors.append("a kept exact-duplicate file is also in another group")
    return errors, components


def inventory():
    files = []
    for name in CLASSES:
        folder = DATA / name
        found = sorted(path.relative_to(ROOT).as_posix() for path in folder.glob("*.jpg"))
        files.extend(found)
    return files


def build_units(files, rows):
    grouped = {}
    for row in rows:
        if row["reviewer_decision"] not in GROUPING:
            continue
        members = tuple(sorted((row["image_a"], row["image_b"])))
        grouped[row["group_id"]] = {
            "group_id": row["group_id"],
            "pair_id": row["pair_id"],
            "decision": row["reviewer_decision"],
            "paths": members,
            "class_name": class_parent(members[0]),
        }
    assigned = {path for group in grouped.values() for path in group["paths"]}
    units_by_class = {name: [] for name in CLASSES}
    for group in grouped.values():
        units_by_class[group["class_name"]].append(
            {"kind": "group", "key": group["paths"][0], "paths": list(group["paths"]), "group": group}
        )
    for rel in files:
        if rel in EXCLUSIONS or rel in assigned:
            continue
        units_by_class[class_parent(rel)].append(
            {"kind": "file", "key": rel, "paths": [rel], "group": None}
        )
    return units_by_class, grouped


def assign_splits(units_by_class):
    assignment = {"train": [], "val": []}
    summary = {}
    placed_groups = []
    for name in CLASSES:
        units = sorted(units_by_class[name], key=lambda unit: unit["key"])
        random.Random(SEED).shuffle(units)
        val_units = int(0.2 * len(units))
        val_part = units[-val_units:] if val_units else []
        train_part = units[:-val_units] if val_units else units
        for side, part in (("train", train_part), ("val", val_part)):
            for unit in part:
                assignment[side].extend(unit["paths"])
                if unit["group"] is not None:
                    placed_groups.append({**unit["group"], "paths": list(unit["group"]["paths"]), "split": side})
        summary[name] = {
            "units": len(units),
            "train_units": len(train_part),
            "val_units": len(val_part),
            "train_files": sum(len(unit["paths"]) for unit in train_part),
            "val_files": sum(len(unit["paths"]) for unit in val_part),
        }
    for side in assignment:
        assignment[side] = sorted(assignment[side])
    return assignment, summary, placed_groups


def hashes_for(paths):
    return {rel: sha256_file(ROOT / rel) for rel in paths}


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    if sys.version_info[:2] != (3, 9):
        raise SystemExit(f"expected CPython 3.9, got {sys.version}")

    fields, rows = load_register()
    apply_remaining_decisions(rows)
    errors, _components = validate(rows)
    if errors:
        raise SystemExit("register validation failed:\n" + "\n".join(errors))
    write_register(fields, rows)

    on_disk = inventory()
    disk_counts = defaultdict(int)
    for rel in on_disk:
        disk_counts[class_parent(rel)] += 1
    missing_excluded = [rel for rel in EXCLUSIONS if rel not in on_disk]
    missing_kept = [rel for rel in KEPT if rel not in on_disk]
    if missing_excluded or missing_kept:
        raise SystemExit(f"missing files excluded={missing_excluded} kept={missing_kept}")

    pool = [rel for rel in on_disk if rel not in EXCLUSIONS]
    digest = hashes_for(on_disk)
    by_hash = defaultdict(list)
    for rel, file_hash in digest.items():
        by_hash[file_hash].append(rel)
    unexpected = []
    for file_hash, members in by_hash.items():
        if len(members) == 1:
            continue
        member_set = set(members)
        if member_set not in (
            {"data/trashNet-dataset/glass/glass176.jpg", "data/trashNet-dataset/plastic/plastic152.jpg"},
            {"data/trashNet-dataset/glass/glass389.jpg", "data/trashNet-dataset/plastic/plastic332.jpg"},
            {"data/trashNet-dataset/glass/glass115.jpg", "data/trashNet-dataset/metal/metal91.jpg"},
        ):
            unexpected.append(sorted(members))
    if unexpected:
        raise SystemExit(f"unexpected exact duplicates, split not written: {unexpected}")

    units_by_class, grouped = build_units(pool, rows)
    covered = []
    for units in units_by_class.values():
        for unit in units:
            covered.extend(unit["paths"])
    if sorted(covered) != sorted(pool):
        raise SystemExit("units do not cover the corrected inventory exactly")

    assignment, summary, placed_groups = assign_splits(units_by_class)
    train, val = assignment["train"], assignment["val"]
    checks = []

    def check(name, ok, detail):
        checks.append({"check": name, "passed": bool(ok), "detail": detail})
        return ok

    ok = True
    ok &= check("no path overlap", set(train).isdisjoint(val), f"overlap {len(set(train) & set(val))}")
    ok &= check(
        "inventory partition",
        sorted(train + val) == sorted(pool),
        f"train {len(train)} val {len(val)} pool {len(pool)}",
    )
    ok &= check(
        "excluded absent",
        all(rel not in train and rel not in val for rel in EXCLUSIONS),
        "three excluded paths are in neither manifest",
    )
    ok &= check(
        "excluded files remain on disk",
        all((ROOT / rel).is_file() for rel in EXCLUSIONS),
        "original files were not deleted",
    )
    train_hashes = {digest[rel] for rel in train}
    val_hashes = {digest[rel] for rel in val}
    ok &= check("no shared SHA-256", train_hashes.isdisjoint(val_hashes), f"shared {len(train_hashes & val_hashes)}")
    combined = train + val
    ok &= check(
        "each SHA-256 once",
        len({digest[rel] for rel in combined}) == len(combined),
        "kept exact-duplicate hashes appear once",
    )
    group_sides = defaultdict(set)
    side_of = {rel: "train" for rel in train}
    side_of.update({rel: "val" for rel in val})
    for group in grouped.values():
        group_sides[group["group_id"]] = {side_of[path] for path in group["paths"]}
    ok &= check(
        "groups stay together",
        all(len(sides) == 1 for sides in group_sides.values()),
        f"{len(group_sides)} groups",
    )
    ok &= check(
        "validation size rule",
        all(summary[name]["val_units"] == int(0.2 * summary[name]["units"]) for name in CLASSES),
        "int(0.2 * units), last units after Random(42)",
    )
    ok &= check("six classes", set(disk_counts) == set(CLASSES), ",".join(CLASSES))
    ok &= check("register complete", True, "139 HUMAN_CONFIRMED rows")
    if not ok:
        failed = [item["check"] for item in checks if not item["passed"]]
        raise SystemExit("split checks failed, manifests not written: " + ", ".join(failed))

    OUT.mkdir(parents=True)
    train_text = "\n".join(train) + "\n"
    val_text = "\n".join(val) + "\n"
    (OUT / "train.txt").write_text(train_text)
    (OUT / "val.txt").write_text(val_text)

    counts_fields = [
        "class_name", "disk_files", "excluded_files", "inventory_files", "units",
        "train_units", "val_units", "train_files", "val_files",
    ]
    with (OUT / "class_counts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=counts_fields, lineterminator="\n")
        writer.writeheader()
        for name in CLASSES:
            excluded_n = sum(1 for rel in EXCLUSIONS if class_parent(rel) == name)
            writer.writerow({
                "class_name": name,
                "disk_files": disk_counts[name],
                "excluded_files": excluded_n,
                "inventory_files": disk_counts[name] - excluded_n,
                "units": summary[name]["units"],
                "train_units": summary[name]["train_units"],
                "val_units": summary[name]["val_units"],
                "train_files": summary[name]["train_files"],
                "val_files": summary[name]["val_files"],
            })

    with (OUT / "exclusions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "folder_label", "reason", "kept_path", "kept_label"],
            lineterminator="\n",
        )
        writer.writeheader()
        pairs = {
            "data/trashNet-dataset/glass/glass176.jpg": ("P001", "data/trashNet-dataset/plastic/plastic152.jpg", "plastic"),
            "data/trashNet-dataset/plastic/plastic332.jpg": ("P002", "data/trashNet-dataset/glass/glass389.jpg", "glass"),
            "data/trashNet-dataset/metal/metal91.jpg": ("P139", "data/trashNet-dataset/glass/glass115.jpg", "glass"),
        }
        for rel in sorted(EXCLUSIONS):
            pair_id, kept, kept_label = pairs[rel]
            writer.writerow({
                "path": rel,
                "folder_label": class_parent(rel),
                "reason": f"{pair_id} conflicting-label exact duplicate; file left on disk and omitted from the inventory",
                "kept_path": kept,
                "kept_label": kept_label,
            })

    with (OUT / "groups.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["group_id", "pair_id", "decision", "class_name", "split", "paths"],
            lineterminator="\n",
        )
        writer.writeheader()
        for group in sorted(placed_groups, key=lambda item: item["group_id"]):
            writer.writerow({
                "group_id": group["group_id"],
                "pair_id": group["pair_id"],
                "decision": group["decision"],
                "class_name": group["class_name"],
                "split": group["split"],
                "paths": "|".join(group["paths"]),
            })

    payload = {
        "experiments": ["EXP-003", "EXP-004"],
        "seed": SEED,
        "rng": "CPython random.Random, constructed again for each class",
        "python": sys.version,
        "rounding": "validation units = int(0.2 * class_units); those are the last units after the shuffle",
        "unit_sort": "lexicographically smallest relative path, then shuffle",
        "manifest_sort": "relative paths sorted lexicographically",
        "disk_counts": {name: disk_counts[name] for name in CLASSES},
        "per_class": summary,
        "exclusions": sorted(EXCLUSIONS),
        "groups": placed_groups,
        "register": str(REGISTER.relative_to(ROOT)),
        "register_sha256": sha256_file(REGISTER),
        "train_txt_sha256": sha256_bytes(train_text.encode()),
        "val_txt_sha256": sha256_bytes(val_text.encode()),
        "train_files": len(train),
        "val_files": len(val),
        "checks": checks,
    }
    (OUT / "split.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"train_files": len(train), "val_files": len(val), "checks": checks, "per_class": summary}, indent=2))


if __name__ == "__main__":
    main()
