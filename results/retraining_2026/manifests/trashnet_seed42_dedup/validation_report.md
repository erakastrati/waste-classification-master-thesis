# Deduplicated TrashNet split

Status: validation passed. Training was not started.

Directory: `results/retraining_2026/manifests/trashnet_seed42_dedup`.

The previous manifests in `trashnet_seed42` were not modified. Source images were not modified.

## Counts

| Class | Inventory | Train | Validation |
|---|---:|---:|---:|
| cardboard | 403 | 323 | 80 |
| glass | 500 | 400 | 100 |
| metal | 408 | 327 | 81 |
| paper | 588 | 471 | 117 |
| plastic | 481 | 385 | 96 |
| trash | 137 | 110 | 27 |
| Total | 2517 | 2016 | 501 |

## Checks

- inventory size: pass. 2517
- train size: pass. 2016
- validation size: pass. 501
- class counts: pass. cardboard 323/80, glass 400/100, metal 327/81, paper 471/117, plastic 385/96, trash 110/27
- path overlap: pass. 0
- SHA-256 overlap: pass. no hash on both sides
- each SHA-256 once: pass. kept exact-duplicate hashes appear once
- P038 same split: pass. train
- no same-object overlap: pass. P038 is the only confirmed same-object group
- review preserved: pass. 0e3227fab6704e597f24c93433671e48f0614dd7629fc37de9ea31cff57a0972
- 139 decisions: pass. P001–P139 HUMAN_CONFIRMED
- no unexpected moves: pass. []
- no unexpected additions: pass. []
- omissions absent: pass. []
- matches previous lists minus omissions: pass. dedup partition equals trashnet_seed42 with seven paths removed
- source files remain: pass. omitted files were not deleted

`train.txt` SHA-256 `bc768a67584e2be3b758a3390de3d90ffb3c4bb90b27e158e422dcf1d468e549`.
`val.txt` SHA-256 `b4b344db8ac6e2cc99199ac12754b8fe951ff8c2cf0d36e14e88bb567a022cb5`.
Review register SHA-256 `0e3227fab6704e597f24c93433671e48f0614dd7629fc37de9ea31cff57a0972`.

No image that remains in the inventory changed side relative to `trashnet_seed42`. The only path differences are the seven omitted visual copies.
