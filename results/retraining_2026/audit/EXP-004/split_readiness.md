# Corrected split readiness

The approved deduplicated lists are `results/retraining_2026/manifests/trashnet_seed42_dedup/`: 2,517 inventory files, 2,016 training, 501 validation. Validation passed with no images moving relative to `trashnet_seed42` except the seven omitted visual copies. `trashnet_seed42/` is the earlier list and was not overwritten. No training script, checkpoint, or source image was changed. EXP-003 and EXP-004 were not run.

The paragraphs below describe `trashnet_seed42` before deduplication.

Manifests: `results/retraining_2026/manifests/trashnet_seed42/train.txt` and `val.txt`. Both EXP-003 and EXP-004 use these lists. No training script, checkpoint, or source image was changed. EXP-003 and EXP-004 were not run.

The earlier stop, when 122 pair IDs were unresolved, is superseded by the completed review. Those IDs are `unrelated` in `visual_review_register.csv`. `unresolved_pair_ids.csv` is that earlier list.

## What the lists do

- Exclude only `glass/glass176.jpg`, `plastic/plastic332.jpg`, and `metal/metal91.jpg` from the inventory. The files remain on disk.
- Keep `plastic/plastic152.jpg`, `glass/glass389.jpg`, and `glass/glass115.jpg` with their existing labels.
- Keep the eight groups in `grouping_report.md` intact. Two paper groups are in validation. The other six groups are in training. None cross the boundary.
- Perceptual components are not groups.
- One `random.Random(42)` shuffle per class, validation size `int(0.2 * units)`.

## Counts

2,527 files on disk, 2,524 in the inventory, 2,021 training, 503 validation. Per-class counts and the check results are in `manifests/trashnet_seed42/class_counts.csv` and `validation_report.md`.

## Still open, and not a split defect

Decision D5, the 30-layer unfreeze, is not settled. Approving these lists does not approve a training run.
