# Dataset mapping audit — TrashNet split used by EXP-003 and EXP-004

Phase 1 inspection only. No training script was executed and `image_dataset_from_directory` was not called again. File bytes for the six exact-duplicate paths were re-hashed. All other membership comes from the saved split lists.

## Sources

| Evidence | Role |
|---|---|
| `src/training/train_efficientnet.py` lines 27–43 | EXP-003 loader arguments. `shuffle` is omitted. |
| `src/training/train_efficientnet_augmented.py` lines 33–49 | EXP-004 uses the same loader arguments. |
| `keras/src/utils/image_dataset_utils.py` lines 26–27, 232–241, 325–327, 388–389 | Installed Keras 3.10.0. Default `shuffle=True`. Seeded shuffle, then the split, then a batch shuffle buffer of `batch_size * 8`. |
| `keras/src/utils/dataset_utils.py` lines 530–535, 613–621, 682–694 | Class folders are sorted. Paths and labels are shuffled with `numpy.random.RandomState(seed)`. Validation is the last `int(0.2 * N)` files. |
| `src/data_audit/check_duplicates.py` `stage_split`, lines 233–259 | Rebuilt the membership with those arguments and `shuffle=True` explicit. |
| `results/duplicate_check/cache/trashnet_split.json` | Saved membership. SHA-256 identical to `results/final_dataset_audit/cache/trashnet_split.json` (both 139,258 bytes). |
| `results/duplicate_check/cache/hash_trashnet.npz` | SHA-256 of file bytes, SHA-256 of decoded RGB pixels, width, height. Decode errors: 0. |
| `results/duplicate_check/cache/index.json` | Path order aligned with the hash arrays. 2,527 TrashNet paths. |
| `results/duplicate_check/supplementary_trashnet_candidates.csv` | Perceptual pairs. Confirmation rule is `check_duplicates.py` lines 97–99 and 374–376: `min(pHash, dHash) <= 10` and ZNCC `>= 0.90`. |
| `results/duplicate_check/supplementary_summary.json` | Aggregate written by `stage_supplementary`, lines 1196–1207. |
| On-disk listing of `data/trashNet-dataset/` | Current class counts. |

Checkpoint metadata says Keras 3.10.0 on 10 June 2026 (EXP-003) and 23 June 2026 (EXP-004). The installed environment checked on 18 November 2026 is also TensorFlow 2.16.2 and Keras 3.10.0 (`results/seed_study/environment.json` is a later seed-study record of the same versions). The June and October interpreters were not both executed in this phase. Membership below is therefore **RECONSTRUCTED** from the October cache, not a byte-level replay of the June process.

## Directory contents

**VERIFIED** by listing `data/trashNet-dataset/` on 18 November 2026:

| Class | Files |
|---|---:|
| cardboard | 403 |
| glass | 501 |
| metal | 410 |
| paper | 594 |
| plastic | 482 |
| trash | 137 |
| Total | 2,527 |

**VERIFIED** from `hash_trashnet.npz` fields `w` and `h`: all 2,527 decoded images are 512×384. That matches `docs/notes.md` lines 33–37. The model does not see 512×384. Both training scripts resize to 224×224 (`image_size=(224, 224)`).

## Split algorithm

**VERIFIED** from the training scripts and the installed Keras 3.10.0 source:

1. Subdirectories are taken in alphanumeric order (`dataset_utils.py` lines 530–535). That order is cardboard, glass, metal, paper, plastic, trash. The same order is the row order of both classification reports.
2. Because `shuffle` is omitted, the default `shuffle=True` applies (`image_dataset_utils.py` line 26).
3. `RandomState(42)` shuffles paths, then a second `RandomState(42)` shuffles labels (`dataset_utils.py` lines 617–621).
4. `num_val_samples = int(0.2 * 2527) = 505`. Training keeps the first 2,022. Validation keeps the last 505 (`dataset_utils.py` lines 682–694).
5. After the membership is fixed, both subsets are wrapped in a `tf.data` shuffle buffer of 256 when `shuffle` stays true (`image_dataset_utils.py` lines 250–253 and 388–389). That buffer changes batch order. It does not move a file from train to validation.
6. No script calls `keras.utils.set_random_seed`. The seed 42 argument is only the split and buffer seed.

`docs/notes.md` lines 39–59 states this 80/20 split and seed 42. It does not state that validation is also buffer-shuffled, and it does not state that the global Python, NumPy, and TensorFlow generators were left unseeded.

## Reconstructed membership

From `trashnet_split.json`:

| List | Files | Unique paths | Overlap with `train_shuffled` |
|---|---:|---:|---:|
| `train_shuffled` | 2,022 | 2,022 | — |
| `val_shuffled` | 505 | 505 | **0 paths** |
| `val_unshuffled_finetune_scripts` | 505 | 505 | **395 paths** |

Class counts in the shuffled lists:

| Class | Train | Validation | Sum | On disk |
|---|---:|---:|---:|---:|
| cardboard | 334 | 69 | 403 | 403 |
| glass | 399 | 102 | 501 | 501 |
| metal | 322 | 88 | 410 | 410 |
| paper | 471 | 123 | 594 | 594 |
| plastic | 393 | 89 | 482 | 482 |
| trash | 103 | 34 | 137 | 137 |

**VERIFIED** that these validation supports are exactly the supports printed in:

- `results/efficientnet/classification_report.txt` lines 5–10
- `results/efficientnet_augmented/classification_report.txt` lines 5–10

Matching supports show that the evaluation which wrote those reports saw the same per-class sizes. They do not, by themselves, prove that every filename matched the October cache. Filename membership remains **RECONSTRUCTED**.

The unshuffled validation list is not used by EXP-003 or EXP-004. It is the list produced with `shuffle=False`: the last 505 files in sorted order, which are 368 plastic and 137 trash. `train_taco_finetune.py` lines 70–78 builds EXP-005 validation with `shuffle=False`. That is a different split from EXP-004. Of those 505 paths, 395 are also in `train_shuffled`. This is the same count reported in `results/duplicate_check/duplicate_check_report.md` lines 105–109.

## Exact byte duplicates inside TrashNet

`hash_trashnet.npz` contains exactly three groups whose file SHA-256 repeats. Each group also has one shared pixel SHA-256, so the files are the same decoded image, not a re-encoded near copy. Width and height are 512×384 in every case. SHA-256 below was recomputed from the current files and matches the cache.

### Pair A — crosses the shuffled split, opposite labels

| Path | Recomputed SHA-256 | Bytes | Shuffled membership |
|---|---|---:|---|
| `data/trashNet-dataset/glass/glass176.jpg` | `c41b99aec8a3257c8668a2144ebbaf8042018f0a85d88e22bf4e413b43693dd8` | 16,192 | validation only |
| `data/trashNet-dataset/plastic/plastic152.jpg` | same | 16,192 | training only |

### Pair B — crosses the shuffled split, opposite labels

| Path | Recomputed SHA-256 | Bytes | Shuffled membership |
|---|---|---:|---|
| `data/trashNet-dataset/glass/glass389.jpg` | `e971f4f8f50e960e454ea724cec922fc2988bbf4027aa48b553e139bb5890968` | 15,420 | training only |
| `data/trashNet-dataset/plastic/plastic332.jpg` | same | 15,420 | validation, and also the unshuffled fine-tune validation list |

### Pair C — both copies are in training, opposite labels

| Path | Recomputed SHA-256 | Bytes | Shuffled membership |
|---|---|---:|---|
| `data/trashNet-dataset/glass/glass115.jpg` | `81546d1362d75fc60718e5852593c58f66f813e8c1e9d1eae6871fedb02e2868` | 8,672 | training only |
| `data/trashNet-dataset/metal/metal91.jpg` | same | 8,672 | training only |

Pair C does not cross train and validation. It still gives one training batch two labels, glass and metal, for one image. It affects both EXP-003 and EXP-004 because both use `train_shuffled`.

The same three groups are the only `phash_dist == 0`, `dhash_dist == 0`, `zncc == 1.0000` rows with `compared_with == trashnet` in `supplementary_trashnet_candidates.csv` (rows for glass176, glass389, and glass115). `supplementary_summary.json` field `trashnet_internal_confirmed_pairs_crossing_train_val` is 138. That 138 is not these two pairs alone.

## What the two-image claim refers to

The claim recorded before this audit is specifically Pair A and Pair B: two validation files that are byte-identical to two training files, with the folder label flipped between glass and plastic. The original rows are:

- `supplementary_trashnet_candidates.csv`: `glass176.jpg` (`val`) confirmed against `plastic152.jpg` (`train`), distances 0/0, ZNCC 1.0000
- the same file: `glass389.jpg` (`train`) confirmed against `plastic332.jpg` (`val+ftval`), distances 0/0, ZNCC 1.0000

They are exact duplicates. They are not near-duplicates and not a train/validation path collision. The path sets are disjoint. The pixels are not.

Both experiments use this shuffled membership, so both images were in the training list once and the validation list once, under opposite labels. **RECONSTRUCTED** from the cached split, as qualified above. Whether a particular epoch's buffer happened to draw them is **UNVERIFIED**; membership in the dataset used by `model.fit` does not depend on the buffer.

The checkpoint files are intact. Pair A and Pair B do not change the Keras zip structure. They change the labels the fit call was given. See the EXP-004 report for the reuse conclusion.

## Other cross-split pairs

Of the 359 confirmed TrashNet-internal pairs, 138 have one path whose split label starts with `train` and one that does not. Strength of those 138, using the minimum of the stored pHash and dHash distances:

| Group | Pairs |
|---|---:|
| Hash distance 0 and ZNCC 1 (Pair A and Pair B) | 2 |
| Minimum hash distance ≤ 4 and ZNCC ≥ 0.98 | 5 |
| Remaining confirmed pairs (minimum distance ≤ 10 and ZNCC ≥ 0.90) | 131 |

Of the 138, 107 stay inside one class name and 31 cross class names. The two exact pairs are part of the 31. The other 136 were not opened visually in this phase. They are a separate contamination question from the two exact validation images. Their effect on the published accuracy was not measured, because this phase did not run inference.

## Follow-up on 18 November 2026

The 138 cross-split rows were recounted. They are 138 unordered pairs and 110 distinct images (73 training, 37 validation). Only the two exact pairs in this file are byte-identical. Five further paper pairs meet a tighter automatic band and have no visual verdict. The remaining pairs, including a 64-image component that is mostly paper, are links from the loose automatic rule. The pair table, the thresholds, and the proposed review order are in `audit/EXP-004/cross_split_pair_review.md` and `audit/EXP-004/cross_split_pairs.csv`. No file under `data/trashNet-dataset/` was removed or relabeled.

## Label index

Folder name is the label. Integer indices follow sorted folder order and are the indices used by `sparse_categorical_crossentropy`:

| Index | Class |
|---:|---|
| 0 | cardboard |
| 1 | glass |
| 2 | metal |
| 3 | paper |
| 4 | plastic |
| 5 | trash |

No class-weight vector is applied by either EfficientNet training script. `class_weight` does not occur in `train_efficientnet.py` or `train_efficientnet_augmented.py`.
