# Dataset labels — expected structure and how to add a new dataset

`datasets/labels/` holds one label file per dataset: `{dataset_name}_labels.csv`
plus any auxiliary provenance maps. Label files are **generated locally** by the
scripts in `label_generation_csv/` (or by your own equivalent); they are not
shipped with this repository, because every dataset remains under its own
license.

## Label CSV schema

| Column | Required | Meaning / convention |
|---|---|---|
| `Name` | yes | image basename (as in `aligned/{dataset}/`) |
| `Path` | yes | path relative to the repository root, e.g. `root_dir/datasets/aligned/{dataset}/img.jpg` |
| `Identity` | yes | subject ID; all images of the same person share the same ID (drives genuine/impostor pairing) |
| `Gender_code` | yes* | `1` = male, `-1` = female, empty = unknown (*needed only for gender evaluations) |
| `Gender` | optional | human-readable gender |
| `Age` | optional | age (years) |
| `Race_code` / `Race` | optional | race label per the source dataset's convention |
| `Emotion_code` | optional | canonical emotion index, see below |
| attribute columns | optional | one 0/1 column per attribute the source data provides (e.g. `Beard`, `Sun glasses`, …) |

Canonical emotion codes (fixed order used across the label scripts and the
evaluations): `0 Neutral, 1 Anger, 2 Scream, 3 Contempt, 4 Disgust, 5 Fear,
6 Happy, 7 Sadness, 8 Surprise`.

**Privacy rule:** committed label files must not contain personal data
(real names, dates of birth, or anything that individually identifies a real
person). Anonymized subject IDs only. If your source labels contain personal
data, strip it before the file leaves your machine.

## Pairs

`datasets/pairs/{dataset}_{impostor|genuine}_pairs.txt`, one pair per line:

```
<identity_A> <image_A> <identity_B> <image_B>
```

`image_A/B` are basenames relative to `aligned/{dataset}/`. These are
generated, not edited by hand: `deid run preprocess` (alignment via MTCNN +
pair generation).

## Adding a new dataset (checklist)

1. **Obtain the dataset** through its official channel and store the raw
   images in `datasets/original/{dataset_name}/` (one folder, images directly
   inside; keep the source license file next to your records, not in the repo).
2. **Align + pairs:** select the dataset and run preprocessing —
   `deid select datasets {name}` and `deid run preprocess`. This writes
   `aligned/{name}/` (256-ish square MTCNN crops) and both pair files.
3. **Labels:** write `{name}_labels.csv` (schema above). Use the matching
   script in `label_generation_csv/` when one exists for your dataset;
   otherwise create the CSV from the source documentation, deriving
   `Identity` / attributes from whatever the dataset guarantees.
4. **Verify:** `deid verify` (or `deid verify --all`) — it checks aligned
   counts, label row/path coverage, empty gender columns, and pair-file
   integrity.
5. **Run:** select technique(s) and evaluation(s), then `deid run all`.

## Repository conventions

- Dataset folder names are lowercase kebab-case (`my-dataset`); the same name
  is used in `original/`, `aligned/`, in labels, in pairs, and in config.
- Technique outputs live in sibling folders `datasets/{Technique}/{dataset}/`
  and are git-ignored (generated data, never committed).
- `deid verify exits 1 on FAIL, 0 otherwise — wire it into CI or a pre-run
  check.
- Nothing under `root_dir/` may be moved, renamed, or deleted by toolkit code
  (the data store is a no-touch zone; the toolkit only adds generated files).
