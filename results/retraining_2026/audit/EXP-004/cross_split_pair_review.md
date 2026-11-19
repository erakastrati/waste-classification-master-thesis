# Cross-split perceptual pairs

Follow-up to the Phase 1 audit. No model was trained or evaluated. No image was moved, deleted, or relabeled. The 136 pairs that are not exact byte copies are not called duplicates here.

## Where the count 138 comes from

The count is written by `stage_supplementary` in `src/data_audit/check_duplicates.py`, lines 1196–1207. It reads `results/duplicate_check/supplementary_trashnet_candidates.csv` and counts rows with `compared_with == trashnet`, `verdict == confirmed`, and split labels that disagree on `startswith("train")`. The stored result is `results/duplicate_check/supplementary_summary.json`, field `trashnet_internal_confirmed_pairs_crossing_train_val`: 138. The same sentence is in `results/duplicate_check/duplicate_check_report.md`, line 102.

Those rows were produced earlier by `stage_perceptual`, lines 448–469, which calls `hash_search` on TrashNet against itself with `same_set=True`. The run log is `results/duplicate_check/log_perceptual.txt`, line 16: 26,226 candidate pairs, 359 with verdict `confirmed`. The 138 are the cross-split subset of those 359. The other 221 confirmed pairs stay on one side of the split.

`hash_search` lines 389–390 keeps only `j > i`. Each unordered pair is stored once. **The 138 are unique pairs, not unique images.**

## Detection method and thresholds

**VERIFIED** from `check_duplicates.py`:

| Step | Implementation | Threshold |
|---|---|---|
| pHash | 32×32 grayscale, DCT, top-left 8×8, 64 bits. `phash_dhash_thumb`, lines 164–172. | Candidate if Hamming distance ≤ 12 (`HASH_T`, line 97). |
| dHash | 9×8 grayscale, 64 bits. Same function. | Same candidate rule: pHash ≤ 12 **or** dHash ≤ 12 (line 393). |
| ZNCC | Zero-mean normalised correlation of the 32×32 thumbnail. `verdict`, lines 373–376. | Automatic verdict `confirmed` if `min(pHash, dHash) ≤ 10` (`CONFIRM_HASH_T`, line 98) and ZNCC ≥ 0.90 (`CONFIRM_ZNCC`, line 99). |
| Pair direction | `same_set=True` blanks indices `j ≤ i` (lines 389–390). | One row per unordered pair. |

The word `confirmed` in the CSV is this automatic rule. It is not a human decision and it is not a SIFT result.

TrashNet hashes have shape `(2527, 1)` in `results/duplicate_check/cache/hash_trashnet.npz` (`ph`, `dh`, and `th`). `stage_hash` line 212 sets the eight dihedral orientations only when `name == "test"`. The TrashNet-versus-TrashNet search therefore compares the original orientation only. It does not search flips or 90-degree rotations. The eight-orientation search described in `duplicate_check_report.md` lines 31–33 applies to the held-out test images, not to these 138 pairs.

SIFT is not part of this count. `stage_sift` and `perceptual_verified.csv` score test-set pairs. None of the 138 rows has a SIFT inlier count.

## What was recounted from the CSV

Recounted in this follow-up from `supplementary_trashnet_candidates.csv`, without rerunning the hash search. The review table is `results/retraining_2026/audit/EXP-004/cross_split_pairs.csv` (138 data rows).

| Quantity | Value |
|---|---:|
| Pairs | 138 |
| Distinct images | 110 |
| Distinct training images | 73 |
| Distinct validation images | 37 |
| Image present on both sides | 0 |
| Images in exactly one pair | 63 |
| Images in more than one pair | 47 |
| Largest number of pairs touching one image | 17 |
| Same folder class | 107 |
| Different folder classes | 31 |
| Connected components | 18 |

Component sizes: one component of 64 images, one of 14, and 16 components of 2. The component of 64 contains 51 paper images, 7 cardboard, 5 plastic, and 1 glass. The component of 14 contains 12 glass, 1 cardboard, and 1 paper. A pair inside either component is a link in a similarity web. It is not, by itself, a separate copy of one photograph.

`supplementary_trashnet_summary.json` also says `confirmed_trashnet_imgs: 120`, `confirmed_val_imgs: 21`, and `confirmed_train_imgs: 94` under `trashnet(internal i<j)`. Those numbers count only the lower file index of each of the 359 pairs, and the validation count keeps only the membership string exactly `val` (`stage_perceptual` lines 456–460). They are not the 110 / 73 / 37 counts above, and they are not limited to the cross-split subset.

Class combinations among the 138 pairs:

| Classes | Pairs |
|---|---:|
| paper–paper | 82 |
| glass–glass | 19 |
| cardboard–paper | 14 |
| paper–plastic | 7 |
| glass–plastic | 3 |
| glass–paper | 3 |
| glass–metal | 2 |
| metal–metal | 2 |
| plastic–plastic | 2 |
| cardboard–cardboard | 1 |
| cardboard–glass | 1 |
| cardboard–plastic | 1 |
| trash–trash | 1 |

The median ZNCC is 0.927. The minimum is 0.9006, which is just above the 0.90 cutoff. Distance of the smaller hash: 4 pairs at 0, then a spread from 1 through 10 (24 pairs at distance 10). Many opposite-class rows pass only because one hash is small while the other is large. Example: `cardboard83.jpg` (train) against `paper110.jpg` (validation) has pHash 22, dHash 0, ZNCC 0.9617. That meets `min(pHash, dHash) ≤ 10`. It is not evidence of the same object.

## Which pairs are actually duplicates

**Byte-identical, opposite labels. Included in the 138. These two pairs only.**

| Pair | Files | Evidence |
|---|---|---|
| P001 in the review CSV | `data/trashNet-dataset/glass/glass176.jpg` (validation, glass) and `data/trashNet-dataset/plastic/plastic152.jpg` (training, plastic) | SHA-256 `c41b99aec8a3257c8668a2144ebbaf8042018f0a85d88e22bf4e413b43693dd8`, 16,192 bytes. pHash 0, dHash 0, ZNCC 1.0000. |
| P002 | `data/trashNet-dataset/glass/glass389.jpg` (training, glass) and `data/trashNet-dataset/plastic/plastic332.jpg` (validation and unshuffled fine-tune validation, plastic) | SHA-256 `e971f4f8f50e960e454ea724cec922fc2988bbf4027aa48b553e139bb5890968`, 15,420 bytes. pHash 0, dHash 0, ZNCC 1.0000. |

The full-dataset SHA-256 grouping in `hash_trashnet.npz` has three repeated groups. The third, `glass115.jpg` and `metal91.jpg`, is inside training on both sides, so it is not one of the 138. No other cross-split pair is an exact byte or exact-pixel duplicate.

**Five same-class pairs pass a tighter automatic band (minimum hash distance ≤ 4 and ZNCC ≥ 0.98) and are not byte-identical.** They are all paper, each in its own component of size 2:

| Files | Distances | ZNCC |
|---|---|---|
| `paper185.jpg` (validation) and `paper390.jpg` (training) | pHash 2, dHash 1 | 0.9953 |
| `paper101.jpg` (training) and `paper108.jpg` (validation) | 2, 2 | 0.9921 |
| `paper180.jpg` (training) and `paper284.jpg` (validation) | 2, 4 | 0.9915 |
| `paper345.jpg` (training) and `paper451.jpg` (validation) | 4, 1 | 0.9913 |
| `paper147.jpg` (validation) and `paper99.jpg` (training) | 2, 2 | 0.9886 |

**UNVERIFIED** whether each of these five is the same object, a recrop, or a different sheet of paper. No contact sheet and no SIFT result exists for them. They are the first pairs a visual review should open. They are not labelled duplicates in this file.

The remaining 131 pairs, including all 29 opposite-class pairs other than the two exact copies, have only the automatic 0.90 / distance-10 rule. That rule accepts different objects. They must not be removed or relabelled on this evidence.

## Visual verification that already exists

`results/duplicate_check/manual_review.csv` has verdicts for held-out test images against Garbage v2, Household, and TACO. It has no TrashNet-internal train/validation row. A search of that file for `glass176`, `plastic152`, `glass389`, and `plastic332` finds nothing.

`results/duplicate_check/review_sheets/` was built by `stage_report` lines 1141–1164 from **test-set** evidence. `log_report.txt` records `near_identical_TrashNet` as 20 test-versus-TrashNet pairs on 3 sheets, and `internal_non_exact` as pairs inside the 600-image test set. Those sheets are not a review of these 138 pairs. `duplicate_check_report.md` lines 112–115 says the visual sample of near-identical TrashNet pairs was 8 test-set examples. It does not say the 138 internal pairs were opened.

Human verification recorded for all 138 rows in the new CSV: `none`. Byte identity of the two exact pairs does not depend on that missing review.

## Review procedure, not yet run

Do not generate sheets or run SIFT until this procedure is approved. It reads images and writes only under `results/retraining_2026/audit/EXP-004/`. It does not change `data/` or the historical audit.

1. Render one contact sheet per priority band, two photographs per row, longest side 400 pixels, with `pair_id` and the two paths printed under the row.
2. Review in this order:
   - the 2 exact pairs, already closed by SHA-256, included so the sheet matches the CSV
   - the 5 paper pairs in the table above (`review_priority` `2_near_automatic`)
   - the 29 opposite-class pairs that are not byte-identical (`3_opposite_class_automatic`)
   - the 6 remaining isolated same-class pairs (`4_isolated_same_class_pair`)
   - a sample of the 64-image component and the 14-image component, not all 96 edges (`5_large_similarity_component`)
3. Record a verdict per `pair_id` in a new CSV: `same_file`, `same_object_different_photo`, `similar_same_class`, or `different`. Leave the TrashNet folders unchanged.
4. Only a `same_file` or `same_object_different_photo` verdict would add a pair to the exclusion list. A missing verdict leaves the pair in the dataset.

Priority counts in `cross_split_pairs.csv`: 2, 5, 29, 6, and 96.
