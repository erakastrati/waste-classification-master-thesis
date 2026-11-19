# Proposal — drop one copy from each visually identical pair

**Applied on 19 November 2026** in `results/retraining_2026/manifests/trashnet_seed42_dedup/`. The original dataset and `trashnet_seed42/` were not modified. The text below is the proposal that was approved.

This asks whether the corrected inventory should contain one file, rather than two, for each pair the reviewer called visually identical. It does not change the decision for P038.

## What the current split already does

The eight confirmed pairs are already atomic units in `trashnet_seed42`. None of them cross training and validation. The within-split effect is that a visually identical pair is stored twice on the same side:

| Pair | Files | Current side | Effect of keeping both |
|---|---|---|---|
| P003 | `paper185.jpg`, `paper390.jpg` | validation | The same paper image is scored twice. |
| P004 | `paper101.jpg`, `paper108.jpg` | training | The same paper image is trained on twice. |
| P005 | `paper180.jpg`, `paper284.jpg` | validation | The same paper image is scored twice. |
| P006 | `paper345.jpg`, `paper451.jpg` | training | The same paper image is trained on twice. |
| P007 | `paper147.jpg`, `paper99.jpg` | training | The same paper image is trained on twice. |
| P037 | `paper2.jpg`, `paper40.jpg` | training | The same paper image is trained on twice. |
| P039 | `metal122.jpg`, `metal57.jpg` | training | The same metal image is trained on twice. |
| P038 | `metal149.jpg`, `metal9.jpg` | training | Two photographs of one object. Both stay. |

These copies are not SHA-256 duplicates. Their file sizes differ, so a “keep the smaller file” rule is a different choice from the rule below.

## Selection rule

For each `visual_duplicate` pair, retain the lexicographically smaller repo-relative path and omit the other path from the corrected inventory. Compare the whole path string, not the integer in the filename. The omitted file stays on disk in its original folder.

This is the same key the current split uses to sort a unit before `random.Random(42)`. Keeping that path leaves every unit key unchanged.

The rule is not a judgment of image quality. It does not use file size, hash distance, or which side the pair sits on.

| Pair | Retain | Omit from the inventory | Why this file |
|---|---|---|---|
| P003 | `paper/paper185.jpg` | `paper/paper390.jpg` | `paper185` < `paper390` |
| P004 | `paper/paper101.jpg` | `paper/paper108.jpg` | `paper101` < `paper108` |
| P005 | `paper/paper180.jpg` | `paper/paper284.jpg` | `paper180` < `paper284` |
| P006 | `paper/paper345.jpg` | `paper/paper451.jpg` | `paper345` < `paper451` |
| P007 | `paper/paper147.jpg` | `paper/paper99.jpg` | `paper147` < `paper99`, because the character `1` precedes `9`. The numeric order 99 < 147 is not the rule. |
| P037 | `paper/paper2.jpg` | `paper/paper40.jpg` | `paper2` < `paper40` |
| P039 | `metal/metal122.jpg` | `metal/metal57.jpg` | `metal122` < `metal57`, because `1` precedes `5`. |

P038 is not in this table. `metal149.jpg` and `metal9.jpg` both remain, and they remain one unit so they cannot be separated.

## Expected counts

“On disk” is the unchanged TrashNet tree. “Inventory” is the corrected list after the three exact-label exclusions already recorded, and, in the proposed column, after also omitting the seven redundant copies.

| Class | On disk | Current inventory | Proposed inventory | Current train / val files | Proposed train / val files |
|---|---:|---:|---:|---:|---:|
| cardboard | 403 | 403 | 403 | 323 / 80 | 323 / 80 |
| glass | 501 | 500 | 500 | 400 / 100 | 400 / 100 |
| metal | 410 | 409 | 408 | 328 / 81 | 327 / 81 |
| paper | 594 | 594 | 588 | 475 / 119 | 471 / 117 |
| plastic | 482 | 481 | 481 | 385 / 96 | 385 / 96 |
| trash | 137 | 137 | 137 | 110 / 27 | 110 / 27 |
| Total | 2,527 | 2,524 | 2,517 | 2,021 / 503 | 2,016 / 501 |

Unit counts do not change. A visually identical pair is already one unit, and the proposal removes the extra file inside that unit. P038 is still the only two-file unit.

| Class | Units | Validation units `int(0.2 × units)` | Proposed train files | Proposed val files |
|---|---:|---:|---:|---:|
| cardboard | 403 | 80 | 323 | 80 |
| glass | 500 | 100 | 400 | 100 |
| metal | 407 | 81 | 327 | 81 |
| paper | 588 | 117 | 471 | 117 |
| plastic | 481 | 96 | 385 | 96 |
| trash | 137 | 27 | 110 | 27 |
| Total | 2,516 | 501 | 2,016 | 501 |

Metal training has 327 files from 326 units because P038 still contributes two photographs. Paper validation falls from 119 files to 117, which is one count per validation image. The two double-counted validation copies are `paper390.jpg` and `paper284.jpg`.

## Regenerate or edit the current lists

Regenerate the split from the deduplicated inventory. Do not hand-edit `trashnet_seed42` in place.

Under this selection rule the two procedures describe the same partition. Each class is still sorted by the lexicographically smallest path and shuffled with a new `random.Random(42)`. After the extra copy is removed, that smallest path is still the unit key, so the shuffle order is the same. The retained file stays on the side where its pair sits now. P038 stays in training as a pair. Every other image stays where it is. Removing the seven omitted paths from the current `train.txt` and `val.txt` would match a fresh run.

That equivalence disappears if the retained file is not the current unit key. Keeping `paper99.jpg` instead of `paper147.jpg`, or `metal57.jpg` instead of `metal122.jpg`, changes the paper or metal sort order. `random.Random(42)` would then assign different images to validation, not only the duplicate. A size-based choice would do the same for several pairs, because the smaller file is not always the lexicographically smaller path.

A later approved run should write a new directory, for example `results/retraining_2026/manifests/trashnet_seed42_dedup/`, and leave `trashnet_seed42/` as the record of the lists before this proposal. The seed, the per-class `int(0.2 × units)` rule, and the shared use by EXP-003 and EXP-004 stay as they are. The historical Keras supports are still not the target.

## Applied

The approved lists are `results/retraining_2026/manifests/trashnet_seed42_dedup/`. `trashnet_seed42/` was left unchanged. The seven omitted files remain in the original TrashNet folders. EXP-003 and EXP-004 were not trained. The choice about unfreezing the last 30 base layers is still open.
