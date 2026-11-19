# Proposed corrected TrashNet split

**Applied on 18 November 2026.** The manifests are `results/retraining_2026/manifests/trashnet_seed42/`. The text below is the procedure that was followed. Its earlier “not ready” statements describe the state before the completed review.

This procedure does not move or relabel any file. EXP-003 and EXP-004 read the same two path lists. Neither experiment calls `image_dataset_from_directory(..., validation_split=0.2)` for the corrected split.

## Why the historical split cannot be reused as the corrected one

The June loader, `src/training/train_efficientnet.py` lines 27–43 and the same arguments in `train_efficientnet_augmented.py` lines 33–49, asks Keras for an 80/20 split with seed 42 and the default shuffle. That shuffle is global, not stratified. Its reconstructed membership is `results/duplicate_check/cache/trashnet_split.json`: 2,022 training paths and 505 validation paths, with these validation supports:

| Class | On disk now | Historical validation |
|---|---:|---:|
| cardboard | 403 | 69 |
| glass | 501 | 102 |
| metal | 410 | 88 |
| paper | 594 | 123 |
| plastic | 482 | 89 |
| trash | 137 | 34 |

Path overlap of those two lists is 0. Content overlap is not 0. Three SHA-256 groups share pixels and carry two folder labels:

| Group | Files | Where they sit |
|---|---|---|
| G1 | `glass/glass176.jpg`, `plastic/plastic152.jpg` | validation and training |
| G2 | `glass/glass389.jpg`, `plastic/plastic332.jpg` | training and validation |
| G3 | `glass/glass115.jpg`, `metal/metal91.jpg` | both in training |

A corrected split has to be an explicit list of paths, so the same files cannot be chosen again by a directory shuffle.

## What is frozen before any shuffle

1. The file inventory is the current `.jpg` files in the six class folders. The count on 18 November 2026 is 2,527.
2. Exact groups are the three SHA-256 groups above. They are facts. What to do with them is not. The register rows are P001, P002, and P139, all `PENDING`.
3. A near-duplicate group exists only after a person writes `same_file` or `same_object_different_photo` in `visual_review_register.csv`. Today that column is empty for all 139 rows. The automatic pHash, dHash, and ZNCC scores do not create a group. The 5 paper pairs, the 29 opposite-class pairs, and the 102 remaining cross-split pairs stay ordinary images until that column is filled.
4. A group with two folder labels is unresolved. The generator must stop rather than pick a label.

Recommended resolution for G1, G2, and G3, still waiting for approval: exclude all six files. Keeping one copy would require choosing glass, plastic, or metal, which is a relabeling decision. Excluding both copies removes the conflict without inventing a label. This recommendation is not applied.

## Split algorithm, once those decisions exist

Use one Python process, CPython 3.9 `random.Random`, not NumPy and not `image_dataset_from_directory`.

1. Build atomic units. An unresolved or excluded file is out of the inventory. A confirmed same-object group becomes one unit and carries the single agreed class. Every other file is a unit of one image.
2. Inside each class, sort units by the lexicographically smallest path in the unit. Shuffle that list with `random.Random(42)`.
3. Validation size for a class is `int(0.2 * number_of_units)`. Validation receives the last that many units. Training receives the rest. A multi-image unit moves entirely to one side.
4. Expand units back to file paths. Sort each manifest before writing it, so the file is stable even though the shuffle order is also stored.
5. Refuse to write the files if any check fails:
   - a path appears in both manifests
   - a SHA-256 appears in both manifests
   - a confirmed same-object group has members on both sides
   - an excluded path appears in either manifest
   - any register row that was supposed to be decided is still `PENDING` among the priority bands named in D8 (the three exact pairs, the five paper pairs, and the 29 opposite-class pairs)
6. Write only under `results/retraining_2026/`, in a new directory `manifests/trashnet_seed42/`, and stop if that directory already exists:
   - `train.txt`, one relative path per line
   - `val.txt`, one relative path per line
   - `split.json`, with the seed, the Python version, the rounding rule, the per-class unit counts, the exclusion list, the group map, the register commit or file hash used, and the SHA-256 of `train.txt` and of `val.txt`

Seed 42 here is not the Keras seed 42. The class margins will not match 69 / 102 / 88 / 123 / 89 / 34. That difference is intended. The old supports came from one global shuffle.

## Class balance

Stratifying by class keeps each class near 20% validation after the approved removals. It does not try to restore the historical supports. Trash remains the small class. Its validation count stays `int(0.2 * remaining_trash_units)`. No class weight is added by this proposal. EXP-003 and EXP-004 historical scripts did not use `class_weight`.

An illustration, not a manifest: if all six exact-duplicate files were excluded and no near-duplicate group were collapsed, the pool would be cardboard 403, glass 498, metal 409, paper 594, plastic 480, trash 137. `int(0.2 * n)` would then reserve 80, 99, 81, 118, 96, and 27 validation files. Those numbers are void if the approval keeps a file or collapses a later group.

## How the two experiments share it

The corrected training entry points, when they are eventually written, take `train.txt` and `val.txt` as their only TrashNet membership. EXP-003 and EXP-004 differ by the four augmentation layers already recorded in the EXP-004 audit, not by the file lists, the seed, the image size, or the batch size. Evaluation uses `val.txt` in the order stored in that file, with no buffer shuffle, so a reported accuracy can be tied to the manifest hash.

The historical checkpoints stay where they are. They are not loaded as the start of the corrected runs.

## What is still blocking

- Reviewer decisions for P001–P139. None are filled in.
- Approval of how G1, G2, and G3 are resolved.
- Approval of this algorithm, including seed 42, per-class `int(0.2 * n)`, and exclusion rather than relabeling.
- A decision on whether the corrected protocol still includes the 30-layer unfreeze. That is D5. It does not change the file lists, but it should be settled in the same decision as the split so the two experiments are specified together.

Until those are accepted, no `train.txt` or `val.txt` is written.
