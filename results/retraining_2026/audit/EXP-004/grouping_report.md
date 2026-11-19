# Confirmed groups

These groups come only from the documented human decisions. Perceptual components C01 and C02 are not groups. The assignment of each group is in `results/retraining_2026/manifests/trashnet_seed42/groups.csv`. GRP-P003 and GRP-P005 are in validation. The other six groups are in training.

## Keep one file, exclude the other

The excluded path is left on disk in its original folder. It is not part of a future corrected inventory. The kept path is an ordinary single-file unit. Its folder already matches the reviewer’s label.

| Pair | Group treatment | Keep | Exclude |
|---|---|---|---|
| P001 | Not a two-file group after the exclusion | `plastic/plastic152.jpg` | `glass/glass176.jpg` |
| P002 | Not a two-file group after the exclusion | `glass/glass389.jpg` | `plastic/plastic332.jpg` |
| P139 | Not a two-file group after the exclusion | `glass/glass115.jpg` | `metal/metal91.jpg` |

## Keep both files in one split

| Group | Decision | Class | Files |
|---|---|---|---|
| GRP-P003 | `visual_duplicate` | paper | `paper185.jpg`, `paper390.jpg` |
| GRP-P004 | `visual_duplicate` | paper | `paper101.jpg`, `paper108.jpg` |
| GRP-P005 | `visual_duplicate` | paper | `paper180.jpg`, `paper284.jpg` |
| GRP-P006 | `visual_duplicate` | paper | `paper345.jpg`, `paper451.jpg` |
| GRP-P007 | `visual_duplicate` | paper | `paper147.jpg`, `paper99.jpg` |
| GRP-P037 | `visual_duplicate` | paper | `paper2.jpg`, `paper40.jpg` |
| GRP-P039 | `visual_duplicate` | metal | `metal122.jpg`, `metal57.jpg` |
| GRP-P038 | `same_object_different_photo` | metal | `metal149.jpg`, `metal9.jpg` |

Each of these eight groups was one unit in its class shuffle. Both paths are in the same manifest, as listed in `groups.csv`. Neither file was removed. The historical split had placed one file of each pair in training and the other in validation.

## Unrelated pairs

P010, P046, P053, P126, P132, and P133 do not create groups. Files in those pairs may still belong to an unresolved pair. Those unresolved pairs are not groups either.
