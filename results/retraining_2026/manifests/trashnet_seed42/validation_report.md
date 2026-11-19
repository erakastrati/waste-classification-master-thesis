# Corrected TrashNet split

Status: **ready for approval of the file lists**. Training has not started. EXP-003 and EXP-004 both use these two lists. The historical checkpoints and scripts were not changed.

Directory: `results/retraining_2026/manifests/trashnet_seed42/`

## Rule

CPython 3.9 `random.Random(42)` is constructed again for each class. Units are sorted by their lexicographically smallest path, shuffled, and the last `int(0.2 * units)` go to validation. A confirmed pair is one unit, so both files land in the same list. Manifests are then sorted. This seed is not the Keras loader seed, and the class counts are not the historical 69 / 102 / 88 / 123 / 89 / 34.

Confirmed units are the seven `visual_duplicate` pairs and the one `same_object_different_photo` pair. The three exact-duplicate conflicts are not two-file units: `glass176.jpg`, `plastic332.jpg`, and `metal91.jpg` are left on disk and omitted. Their partners stay, with the labels they already have.

## Counts

| Class | On disk | Excluded | Inventory | Units | Train files | Val files |
|---|---:|---:|---:|---:|---:|---:|
| cardboard | 403 | 0 | 403 | 403 | 323 | 80 |
| glass | 501 | 1 | 500 | 500 | 400 | 100 |
| metal | 410 | 1 | 409 | 407 | 328 | 81 |
| paper | 594 | 0 | 594 | 588 | 475 | 119 |
| plastic | 482 | 1 | 481 | 481 | 385 | 96 |
| trash | 137 | 0 | 137 | 137 | 110 | 27 |
| Total | 2,527 | 3 | 2,524 | 2,516 | 2,021 | 503 |

Metal has two two-file units, both in training, so training has 328 files from 326 units. Paper has six two-file units: four in training and two in validation (`GRP-P003`, `GRP-P005`), so validation has 119 files from 117 units.

## Groups

| Group | Split |
|---|---|
| GRP-P003 paper185.jpg, paper390.jpg | validation |
| GRP-P004 paper101.jpg, paper108.jpg | training |
| GRP-P005 paper180.jpg, paper284.jpg | validation |
| GRP-P006 paper345.jpg, paper451.jpg | training |
| GRP-P007 paper147.jpg, paper99.jpg | training |
| GRP-P037 paper2.jpg, paper40.jpg | training |
| GRP-P038 metal149.jpg, metal9.jpg | training |
| GRP-P039 metal122.jpg, metal57.jpg | training |

## Checks

All passed.

- 139 pair IDs, each `HUMAN_CONFIRMED`, none missing.
- No image path belongs to more than one keep-together group.
- No `unrelated` pair connects two files that were also grouped together.
- Train and validation share no path and no SHA-256.
- Every inventory file is in exactly one list. Each SHA-256 appears once.
- The three excluded paths are in neither list and still exist under `data/trashNet-dataset/`.
- No exact duplicate other than the three reviewed pairs was found.

`split.json` records SHA-256 `89380f2fc59ee5063a0383903225779b7b3659528376815941313197d5cf42ea` for `train.txt` and `b6f18ad2baec2b11ee4abd552af043037dfd16015c502eebb5a15b453d1b35c2` for `val.txt`. Both match the files on disk. The register digest in that file is `0e3227fab6704e597f24c93433671e48f0614dd7629fc37de9ea31cff57a0972`.

## Not part of this approval

Decision D5, whether a corrected run unfreezes the last 30 base layers, is still open. It does not change these lists. Approving the split does not start training.
