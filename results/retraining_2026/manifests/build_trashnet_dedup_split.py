"""Write the approved deduplicated TrashNet split into a new directory.

Does not modify data/trashNet-dataset, the review register, or
manifests/trashnet_seed42. Does not train.
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
PREVIOUS = ROOT / "results/retraining_2026/manifests/trashnet_seed42"
OUT = ROOT / "results/retraining_2026/manifests/trashnet_seed42_dedup"
DATA = ROOT / "data/trashNet-dataset"
CLASSES = ("cardboard", "glass", "metal", "paper", "plastic", "trash")
SEED = 42
EXPECTED_REGISTER_SHA256 = "0e3227fab6704e597f24c93433671e48f0614dd7629fc37de9ea31cff57a0972"

LABEL_EXCLUSIONS = {
    "data/trashNet-dataset/glass/glass176.jpg": ("P001", "data/trashNet-dataset/plastic/plastic152.jpg", "plastic"),
    "data/trashNet-dataset/plastic/plastic332.jpg": ("P002", "data/trashNet-dataset/glass/glass389.jpg", "glass"),
    "data/trashNet-dataset/metal/metal91.jpg": ("P139", "data/trashNet-dataset/glass/glass115.jpg", "glass"),
}
VISUAL_OMISSIONS = {
    "P003": ("data/trashNet-dataset/paper/paper185.jpg", "data/trashNet-dataset/paper/paper390.jpg"),
    "P004": ("data/trashNet-dataset/paper/paper101.jpg", "data/trashNet-dataset/paper/paper108.jpg"),
    "P005": ("data/trashNet-dataset/paper/paper180.jpg", "data/trashNet-dataset/paper/paper284.jpg"),
    "P006": ("data/trashNet-dataset/paper/paper345.jpg", "data/trashNet-dataset/paper/paper451.jpg"),
    "P007": ("data/trashNet-dataset/paper/paper147.jpg", "data/trashNet-dataset/paper/paper99.jpg"),
    "P037": ("data/trashNet-dataset/paper/paper2.jpg", "data/trashNet-dataset/paper/paper40.jpg"),
    "P039": ("data/trashNet-dataset/metal/metal122.jpg", "data/trashNet-dataset/metal/metal57.jpg"),
}
P038 = (
    "data/trashNet-dataset/metal/metal149.jpg",
    "data/trashNet-dataset/metal/metal9.jpg",
)
EXPECTED_FILES = {
    "cardboard": (403, 323, 80),
    "glass": (500, 400, 100),
    "metal": (408, 327, 81),
    "paper": (588, 471, 117),
    "plastic": (481, 385, 96),
    "trash": (137, 110, 27),
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def class_name(rel):
    return rel.split("/")[2]


def read_list(path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_register():
    with REGISTER.open(newline="") as handle:
        return list(csv.DictReader(handle))


def inventory():
    files = []
    for name in CLASSES:
        files.extend(sorted(path.relative_to(ROOT).as_posix() for path in (DATA / name).glob("*.jpg")))
    return files


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    if sys.version_info[:2] != (3, 9):
        raise SystemExit(f"expected CPython 3.9, got {sys.version}")
    register_hash = sha256_file(REGISTER)
    if register_hash != EXPECTED_REGISTER_SHA256:
        raise SystemExit("review register hash changed; decisions were not left untouched")

    rows = load_register()
    decisions = {}
    for row in rows:
        decisions[row["pair_id"]] = row["reviewer_decision"]
    expected_ids = [f"P{n:03d}" for n in range(1, 140)]
    if [row["pair_id"] for row in rows] != expected_ids:
        raise SystemExit("register pair IDs are not P001–P139 in order")
    if any(row["visual_review_status"] != "HUMAN_CONFIRMED" for row in rows):
        raise SystemExit("a review decision is no longer HUMAN_CONFIRMED")
    for pair_id, (retain, omit) in VISUAL_OMISSIONS.items():
        if decisions[pair_id] != "visual_duplicate":
            raise SystemExit(f"{pair_id} is not visual_duplicate")
        if tuple(sorted((retain, omit))) != tuple(sorted((
            next(row["image_a"] for row in rows if row["pair_id"] == pair_id),
            next(row["image_b"] for row in rows if row["pair_id"] == pair_id),
        ))):
            raise SystemExit(f"{pair_id} paths do not match the register")
        if retain != min(retain, omit):
            raise SystemExit(f"{pair_id} retain path is not the lexicographic minimum")
    if decisions["P038"] != "same_object_different_photo":
        raise SystemExit("P038 decision changed")
    p038_row = next(row for row in rows if row["pair_id"] == "P038")
    if set(P038) != {p038_row["image_a"], p038_row["image_b"]}:
        raise SystemExit("P038 paths do not match the register")

    on_disk = inventory()
    disk_counts = defaultdict(int)
    for rel in on_disk:
        disk_counts[class_name(rel)] += 1
    label_omit = set(LABEL_EXCLUSIONS)
    visual_omit = {omit for _retain, omit in VISUAL_OMISSIONS.values()}
    visual_keep = {retain for retain, _omit in VISUAL_OMISSIONS.values()}
    if label_omit & visual_omit:
        raise SystemExit("a label exclusion is also a visual omission")
    missing = [rel for rel in label_omit | visual_omit | visual_keep | set(P038) if rel not in on_disk]
    if missing:
        raise SystemExit(f"missing files: {missing}")

    pool = [rel for rel in on_disk if rel not in label_omit and rel not in visual_omit]
    digest = {rel: sha256_file(ROOT / rel) for rel in on_disk}
    by_hash = defaultdict(list)
    for rel, file_hash in digest.items():
        by_hash[file_hash].append(rel)
    known_exact = [
        {"data/trashNet-dataset/glass/glass176.jpg", "data/trashNet-dataset/plastic/plastic152.jpg"},
        {"data/trashNet-dataset/glass/glass389.jpg", "data/trashNet-dataset/plastic/plastic332.jpg"},
        {"data/trashNet-dataset/glass/glass115.jpg", "data/trashNet-dataset/metal/metal91.jpg"},
    ]
    for members in by_hash.values():
        if len(members) > 1 and set(members) not in known_exact:
            raise SystemExit(f"unexpected exact duplicate: {sorted(members)}")

    units_by_class = {name: [] for name in CLASSES}
    p038 = tuple(sorted(P038))
    units_by_class["metal"].append({"key": p038[0], "paths": list(p038), "group_id": "GRP-P038"})
    assigned = set(p038)
    for rel in pool:
        if rel in assigned:
            continue
        units_by_class[class_name(rel)].append({"key": rel, "paths": [rel], "group_id": ""})

    train, val = [], []
    summary = {}
    for name in CLASSES:
        units = sorted(units_by_class[name], key=lambda unit: unit["key"])
        random.Random(SEED).shuffle(units)
        val_n = int(0.2 * len(units))
        val_part = units[-val_n:] if val_n else []
        train_part = units[:-val_n] if val_n else units
        train.extend(path for unit in train_part for path in unit["paths"])
        val.extend(path for unit in val_part for path in unit["paths"])
        summary[name] = {
            "disk_files": disk_counts[name],
            "inventory_files": sum(len(unit["paths"]) for unit in units),
            "units": len(units),
            "train_units": len(train_part),
            "val_units": len(val_part),
            "train_files": sum(len(unit["paths"]) for unit in train_part),
            "val_files": sum(len(unit["paths"]) for unit in val_part),
        }
    train = sorted(train)
    val = sorted(val)

    previous_train = set(read_list(PREVIOUS / "train.txt"))
    previous_val = set(read_list(PREVIOUS / "val.txt"))
    moved = []
    for rel in train:
        if rel in previous_val:
            moved.append((rel, "validation", "training"))
    for rel in val:
        if rel in previous_train:
            moved.append((rel, "training", "validation"))
    unexpected_new = sorted((set(train) | set(val)) - previous_train - previous_val)
    not_removed = sorted((visual_omit | label_omit) & (set(train) | set(val)))

    checks = []

    def check(name, ok, detail):
        checks.append({"check": name, "passed": bool(ok), "detail": detail})
        return bool(ok)

    ok = True
    ok &= check("inventory size", len(pool) == 2517, str(len(pool)))
    ok &= check("train size", len(train) == 2016, str(len(train)))
    ok &= check("validation size", len(val) == 501, str(len(val)))
    ok &= check(
        "class counts",
        all(
            (summary[name]["inventory_files"], summary[name]["train_files"], summary[name]["val_files"]) == expected
            for name, expected in EXPECTED_FILES.items()
        ),
        "cardboard 323/80, glass 400/100, metal 327/81, paper 471/117, plastic 385/96, trash 110/27",
    )
    ok &= check("path overlap", set(train).isdisjoint(val), str(len(set(train) & set(val))))
    ok &= check(
        "SHA-256 overlap",
        {digest[rel] for rel in train}.isdisjoint({digest[rel] for rel in val}),
        "no hash on both sides",
    )
    ok &= check(
        "each SHA-256 once",
        len({digest[rel] for rel in train + val}) == len(train) + len(val),
        "kept exact-duplicate hashes appear once",
    )
    p038_sides = {("train" if rel in train else "val") for rel in P038}
    ok &= check("P038 same split", p038_sides == {"train"}, ",".join(sorted(p038_sides)))
    ok &= check("no same-object overlap", p038_sides != {"train", "val"}, "P038 is the only confirmed same-object group")
    ok &= check("review preserved", register_hash == EXPECTED_REGISTER_SHA256, register_hash)
    ok &= check("139 decisions", len(rows) == 139, "P001–P139 HUMAN_CONFIRMED")
    ok &= check("no unexpected moves", moved == [], str(moved))
    ok &= check("no unexpected additions", unexpected_new == [], str(unexpected_new))
    ok &= check("omissions absent", not_removed == [], str(not_removed))
    ok &= check(
        "matches previous lists minus omissions",
        set(train) == previous_train - visual_omit and set(val) == previous_val - visual_omit,
        "dedup partition equals trashnet_seed42 with seven paths removed",
    )
    ok &= check(
        "source files remain",
        all((ROOT / rel).is_file() for rel in label_omit | visual_omit),
        "omitted files were not deleted",
    )
    if not ok:
        failed = [item for item in checks if not item["passed"]]
        raise SystemExit("checks failed; no directory written:\n" + json.dumps(failed, indent=2))

    OUT.mkdir(parents=True)
    train_text = "\n".join(train) + "\n"
    val_text = "\n".join(val) + "\n"
    (OUT / "train.txt").write_text(train_text)
    (OUT / "val.txt").write_text(val_text)

    with (OUT / "class_counts.csv").open("w", newline="") as handle:
        fields = ["class_name", "disk_files", "label_exclusions", "visual_omissions", "inventory_files", "units", "train_units", "val_units", "train_files", "val_files"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for name in CLASSES:
            writer.writerow({
                "class_name": name,
                "disk_files": summary[name]["disk_files"],
                "label_exclusions": sum(1 for rel in label_omit if class_name(rel) == name),
                "visual_omissions": sum(1 for rel in visual_omit if class_name(rel) == name),
                "inventory_files": summary[name]["inventory_files"],
                "units": summary[name]["units"],
                "train_units": summary[name]["train_units"],
                "val_units": summary[name]["val_units"],
                "train_files": summary[name]["train_files"],
                "val_files": summary[name]["val_files"],
            })

    with (OUT / "exclusions.csv").open("w", newline="") as handle:
        fields = ["path", "reason", "pair_id", "retained_path", "on_disk"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for rel, (pair_id, kept, _label) in sorted(LABEL_EXCLUSIONS.items()):
            writer.writerow({
                "path": rel,
                "reason": "conflicting-label exact duplicate",
                "pair_id": pair_id,
                "retained_path": kept,
                "on_disk": "yes",
            })
        for pair_id, (retain, omit) in VISUAL_OMISSIONS.items():
            writer.writerow({
                "path": omit,
                "reason": "redundant visually identical copy; lexicographically larger path",
                "pair_id": pair_id,
                "retained_path": retain,
                "on_disk": "yes",
            })

    with (OUT / "groups.csv").open("w", newline="") as handle:
        fields = ["group_id", "pair_id", "decision", "class_name", "split", "paths"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerow({
            "group_id": "GRP-P038",
            "pair_id": "P038",
            "decision": "same_object_different_photo",
            "class_name": "metal",
            "split": "train",
            "paths": "|".join(p038),
        })
        for pair_id, (retain, omit) in VISUAL_OMISSIONS.items():
            side = "train" if retain in train else "val"
            writer.writerow({
                "group_id": f"GRP-{pair_id}",
                "pair_id": pair_id,
                "decision": "visual_duplicate_one_retained",
                "class_name": class_name(retain),
                "split": side,
                "paths": retain,
            })

    payload = {
        "experiments": ["EXP-003", "EXP-004"],
        "directory": "results/retraining_2026/manifests/trashnet_seed42_dedup",
        "previous_manifest": "results/retraining_2026/manifests/trashnet_seed42",
        "seed": SEED,
        "rng": "CPython random.Random, constructed again for each class",
        "python": sys.version,
        "rounding": "validation units = int(0.2 * class_units); those are the last units after the shuffle",
        "selection_rule": "for each visual_duplicate pair, retain the lexicographically smaller repo-relative path",
        "label_exclusions": sorted(label_omit),
        "visual_omissions": sorted(visual_omit),
        "p038_paths": list(p038),
        "per_class": summary,
        "inventory_files": len(pool),
        "train_files": len(train),
        "val_files": len(val),
        "register_sha256": register_hash,
        "train_txt_sha256": sha256_bytes(train_text.encode()),
        "val_txt_sha256": sha256_bytes(val_text.encode()),
        "previous_train_txt_sha256": sha256_file(PREVIOUS / "train.txt"),
        "previous_val_txt_sha256": sha256_file(PREVIOUS / "val.txt"),
        "movement": moved,
        "checks": checks,
    }
    (OUT / "split.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# Deduplicated TrashNet split",
        "",
        "Status: validation passed. Training was not started.",
        "",
        f"Directory: `{payload['directory']}`.",
        "",
        "The previous manifests in `trashnet_seed42` were not modified. Source images were not modified.",
        "",
        "## Counts",
        "",
        "| Class | Inventory | Train | Validation |",
        "|---|---:|---:|---:|",
    ]
    for name in CLASSES:
        item = summary[name]
        lines.append(f"| {name} | {item['inventory_files']} | {item['train_files']} | {item['val_files']} |")
    lines.extend([
        f"| Total | {len(pool)} | {len(train)} | {len(val)} |",
        "",
        "## Checks",
        "",
    ])
    for item in checks:
        state = "pass" if item["passed"] else "fail"
        lines.append(f"- {item['check']}: {state}. {item['detail']}")
    lines.extend([
        "",
        f"`train.txt` SHA-256 `{payload['train_txt_sha256']}`.",
        f"`val.txt` SHA-256 `{payload['val_txt_sha256']}`.",
        f"Review register SHA-256 `{register_hash}`.",
        "",
        "No image that remains in the inventory changed side relative to `trashnet_seed42`. The only path differences are the seven omitted visual copies.",
        "",
    ])
    (OUT / "validation_report.md").write_text("\n".join(lines))
    print(json.dumps({"out": str(OUT.relative_to(ROOT)), "train": len(train), "val": len(val), "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
