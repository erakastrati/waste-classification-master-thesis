# Visual review sheets

Generated from `cross_split_pairs.csv` plus the within-training exact pair. Source files under `data/trashNet-dataset/` were only read. Every `visual_review_status` in `../visual_review_register.csv` is `PENDING`. A similar photograph on a sheet is not a duplicate decision.

Display copies are scaled with Lanczos so the sheet fits a page. The pixels in `data/` are unchanged.

Two pairs are drawn on each sheet, except where a band has an odd remainder. Each pair shows the pair ID, component, pHash, dHash, minimum hash distance, ZNCC, byte-identical flag, filename, class, and train or validation membership.

| Band | Pairs | Sheets |
|---|---:|---:|
| `01_exact_cross_split` | 2 | 1 |
| `02_exact_within_training` | 1 | 1 |
| `03_high_similarity_paper` | 5 | 3 |
| `04_opposite_class` | 29 | 15 |
| `05_by_component/C01` | 77 | 39 |
| `05_by_component/C02` | 19 | 10 |
| `05_by_component/C08` | 1 | 1 |
| `05_by_component/C09` | 1 | 1 |
| `05_by_component/C14` | 1 | 1 |
| `05_by_component/C16` | 1 | 1 |
| `05_by_component/C17` | 1 | 1 |
| `05_by_component/C18` | 1 | 1 |
| Total | 139 | 75 |

The 138 cross-split pairs are P001–P138. P139 is `glass115.jpg` and `metal91.jpg`, both in training. It is not one of the 138. No image file was missing.

`sheet_index.csv` maps every pair ID to its sheet. The register is `../visual_review_register.csv`.
