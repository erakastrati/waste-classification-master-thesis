# Images that appear in more than one pair

`repeated_images.csv` lists every TrashNet path that occurs in two or more of the 139 pairs. There are **47** such paths. The most frequent is `data/trashNet-dataset/paper/paper110.jpg`, in 17 pairs (PDF pages include 6, 8, and 33–64). Next are `paper385.jpg` and `paper477.jpg`, 12 pairs each.

The six SHA-256 files in the exclusion proposal each occur in only one pair. The ten paper files in P003–P007 also occur in only one pair. A decision on those pairs does not have to be copied onto a second row. A decision on `paper110.jpg`, `cardboard83.jpg`, or `glass15.jpg` does.

## Consistency rule

One path has one fate in the corrected inventory.

- If any pair proposes exclusion of a path and that proposal is approved, the path is absent from every remaining pair. It cannot be kept by another row.
- If two pairs are both decided `same_file` or `same_object_different_photo` and they share a path, all of the files named by those pairs form one group. The group cannot be split across training and validation.
- A `different` decision applies to that pair only. The same file may still be the same object as a third file in another pair.
- `other_pair_ids` in `pending_pairs_review_table.csv` is the list to check before a row is signed.

The automatic component IDs C01 and C02 are not decisions. They only show which pending rows share the perceptual graph. Sharing a component does not mean the photographs are the same object.
