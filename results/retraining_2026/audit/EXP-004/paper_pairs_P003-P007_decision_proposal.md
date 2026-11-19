# Decision proposal — paper pairs P003–P007

Status: **undecided**. All five rows in `visual_review_register.csv` remain `PENDING`. The similarity scores are not a decision, and they are not human approval.

These five pairs are paper against paper. None is a SHA-256 duplicate (`exact_duplicate` is `no`). Each of the ten files appears in only this one pair, so a later decision on the pair does not have to be reconciled with a second pair.

| Pair | PDF page | Sheet | Training file | Validation file | pHash | dHash | ZNCC |
|---|---:|---|---|---|---:|---:|---:|
| P003 | 3 | `visual_review/03_high_similarity_paper/sheet_01.png` | `paper/paper390.jpg` | `paper/paper185.jpg` | 2 | 1 | 0.9953 |
| P004 | 3 | `visual_review/03_high_similarity_paper/sheet_01.png` | `paper/paper101.jpg` | `paper/paper108.jpg` | 2 | 2 | 0.9921 |
| P005 | 4 | `visual_review/03_high_similarity_paper/sheet_02.png` | `paper/paper180.jpg` | `paper/paper284.jpg` | 2 | 4 | 0.9915 |
| P006 | 4 | `visual_review/03_high_similarity_paper/sheet_02.png` | `paper/paper345.jpg` | `paper/paper451.jpg` | 4 | 1 | 0.9913 |
| P007 | 5 | `visual_review/03_high_similarity_paper/sheet_03.png` | `paper/paper99.jpg` | `paper/paper147.jpg` | 2 | 2 | 0.9886 |

Paths are under `data/trashNet-dataset/`. P003 and P007 have the validation file listed first in the register; the table above names the training file and the validation file directly.

## Decision required for each pair

One of the following, written into `reviewer_decision` only after a person looks at the sheet:

| Decision | Later effect, if the split policy is approved |
|---|---|
| `different` | Both files stay as ordinary paper images and may fall on opposite sides of the split. |
| `same_file` or `same_object_different_photo` | The two files become one same-object group. The group is assigned entirely to training or entirely to validation. |
| `exclude_both` | Both files leave the corrected inventory. |

No option is selected here. A high ZNCC does not choose among them.

## Why this blocks a leakage-free split

Each pair already has one file in the historical training list and one in the historical validation list. If a pair is later judged to be the same object, the historical split leaked that object. The corrected split cannot be called free of same-object leakage while these five decisions are empty.
