# EXP-004 run directory

Reserved for a future corrected run of EfficientNetB0 with augmentation. Phase 1 did not create a run and did not copy the historical checkpoint.

The historical checkpoint remains at `results/efficientnet_augmented/efficientnet_augmented.keras`. It is not a starting file to be overwritten.

## Unique run folders

Create one new directory per attempt:

`runs/<UTC timestamp>_<short label>/`

Example: `runs/20261118T210000Z_phase1-frozen-head/`

If the chosen directory already exists, stop. Do not delete, rename, or replace files inside an existing run directory. Do not write checkpoints, logs, or reports into this directory itself.
