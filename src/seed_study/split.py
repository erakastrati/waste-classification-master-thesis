"""Frozen TrashNet Train/Validation membership.

The lists are copied from the previously verified seed=42 Keras split.
Nothing in this module reshuffles the dataset, and the training seed is not
an argument.
"""

import json
from collections import Counter
from pathlib import Path

from src.seed_study.protocol import (
    CLASS_NAMES,
    EXPECTED_TRAIN_COUNTS,
    EXPECTED_TRAIN_N,
    EXPECTED_VAL_COUNTS,
    EXPECTED_VAL_N,
    MANIFEST_DIR,
    PROJECT_ROOT,
    SOURCE_SPLIT_JSONS,
    SPLIT_SEED,
)

IMAGE_EXTENSIONS = (".bmp", ".gif", ".jpeg", ".jpg", ".png")
TRAIN_LIST = MANIFEST_DIR / "train_paths.txt"
VAL_LIST = MANIFEST_DIR / "val_paths.txt"
SPLIT_INFO = MANIFEST_DIR / "split_info.json"


def class_name(path):
    return Path(path).parent.name


def class_counts(paths):
    return dict(sorted(Counter(class_name(path) for path in paths).items()))


def read_lines(path):
    text = Path(path).read_text(encoding="utf-8")
    lines = text.splitlines()
    if any(line == "" for line in lines):
        raise ValueError(f"{path} contains a blank line")
    return lines


def write_lines(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_source_lists():
    loaded = []
    for path in SOURCE_SPLIT_JSONS:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        loaded.append(
            (path, list(payload["train_shuffled"]), list(payload["val_shuffled"]))
        )
    reference = loaded[0]
    for path, train_paths, val_paths in loaded[1:]:
        if train_paths != reference[1] or val_paths != reference[2]:
            raise RuntimeError(
                f"Verified split files disagree: {reference[0]} and {path}"
            )
    return reference[1], reference[2]


def freeze_manifest():
    """Write the manifest once. An existing manifest is checked and kept."""
    train_paths, val_paths = load_source_lists()
    verify_membership(train_paths, val_paths)
    present = [path for path in (TRAIN_LIST, VAL_LIST, SPLIT_INFO) if path.exists()]
    if present and len(present) != 3:
        raise RuntimeError(
            "The split manifest is incomplete and was not overwritten: "
            + ", ".join(str(path) for path in present)
        )
    if len(present) == 3:
        existing_train, existing_val = load_frozen_split()
        if existing_train != train_paths or existing_val != val_paths:
            raise RuntimeError(
                "results/seed_study/manifest does not match the verified "
                "seed=42 path lists. The manifest was not overwritten."
            )
        return train_paths, val_paths
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    write_lines(TRAIN_LIST, train_paths)
    write_lines(VAL_LIST, val_paths)
    info = {
        "split_seed": SPLIT_SEED,
        "training_seeds_do_not_change_membership": True,
        "source": [str(path.relative_to(PROJECT_ROOT)) for path in SOURCE_SPLIT_JSONS],
        "n_train": len(train_paths),
        "n_val": len(val_paths),
        "class_names": list(CLASS_NAMES),
        "train_class_counts": class_counts(train_paths),
        "val_class_counts": class_counts(val_paths),
        "overlap": 0,
        "metric_set": "VALIDATION",
        "note": (
            "Copied from the verified Keras validation_split=0.2 shuffle seed=42 "
            "lists. Do not regenerate this membership with training seeds "
            "42, 123, or 456."
        ),
    }
    SPLIT_INFO.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return train_paths, val_paths


def load_frozen_split():
    """Return the frozen lists. This function accepts no seed."""
    if not TRAIN_LIST.exists() or not VAL_LIST.exists():
        raise FileNotFoundError(
            "Frozen split manifest is missing. Run python -m src.seed_study.preflight."
        )
    return read_lines(TRAIN_LIST), read_lines(VAL_LIST)


def verify_membership(train_paths, val_paths):
    errors = []
    if len(train_paths) != EXPECTED_TRAIN_N or len(set(train_paths)) != EXPECTED_TRAIN_N:
        errors.append(
            f"train paths: count={len(train_paths)} unique={len(set(train_paths))} "
            f"expected={EXPECTED_TRAIN_N}"
        )
    if len(val_paths) != EXPECTED_VAL_N or len(set(val_paths)) != EXPECTED_VAL_N:
        errors.append(
            f"validation paths: count={len(val_paths)} unique={len(set(val_paths))} "
            f"expected={EXPECTED_VAL_N}"
        )
    overlap = set(train_paths) & set(val_paths)
    if overlap:
        errors.append(f"train/validation overlap: {len(overlap)}")
    if class_counts(train_paths) != EXPECTED_TRAIN_COUNTS:
        errors.append(f"train class counts {class_counts(train_paths)}")
    if class_counts(val_paths) != EXPECTED_VAL_COUNTS:
        errors.append(f"validation class counts {class_counts(val_paths)}")
    unknown = sorted(
        {class_name(path) for path in train_paths + val_paths} - set(CLASS_NAMES)
    )
    if unknown:
        errors.append(f"unexpected class names: {unknown}")
    if errors:
        raise RuntimeError("Frozen split check failed: " + "; ".join(errors))
    return True


def resolved_path(relative_path):
    path = Path(relative_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def verify_files_exist(train_paths, val_paths):
    """Confirm the frozen lists are exactly the current TrashNet image files."""
    missing = [
        path for path in train_paths + val_paths if not resolved_path(path).is_file()
    ]
    dataset_root = PROJECT_ROOT / "data" / "trashNet-dataset"
    on_disk = []
    other = []
    for directory, dirnames, filenames in __import__("os").walk(dataset_root):
        dirnames[:] = sorted(name for name in dirnames if not name.startswith("."))
        for name in sorted(filenames):
            if name.startswith("."):
                continue
            full = Path(directory) / name
            if name.lower().endswith(IMAGE_EXTENSIONS):
                on_disk.append(full.resolve())
            else:
                other.append(str(full))
    listed = {resolved_path(path) for path in train_paths + val_paths}
    disk_set = set(on_disk)
    errors = []
    if missing:
        errors.append(f"{len(missing)} manifest paths are missing on disk")
    if listed - disk_set:
        errors.append(f"{len(listed - disk_set)} manifest paths are not current dataset files")
    if disk_set - listed:
        errors.append(f"{len(disk_set - listed)} current TrashNet images are outside the frozen split")
    if other:
        errors.append(f"{len(other)} non-image files are in the TrashNet directory")
    if len(disk_set) != EXPECTED_TRAIN_N + EXPECTED_VAL_N:
        errors.append(f"disk image count is {len(disk_set)}")
    if errors:
        raise RuntimeError("TrashNet file check failed: " + "; ".join(errors))
    return True
