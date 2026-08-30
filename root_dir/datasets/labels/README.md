# KDEF naming and provenance

Three naming schemes are in play for KDEF in this repository.

## 1. Our aligned dataset (what everything is keyed on)

`root_dir/datasets/aligned/kdef/<EMOTION>_<actor>_<shot>.jpg`, e.g. `ANG_0_25.jpg`

- `<EMOTION>`: `NEU,ANG,SAD,DIS,FEA,HAP,SUR,CON` (8 codes)
- `<actor>`: 0–139 — our sequential actor ID (**the canonical key** for all
  labels, embeddings and analysis)
- `<shot>`: shot index within (actor, emotion) in the source release
  `datasets/original/KDEF/<emotion>/<actor>_<shot>.jpg` (the 2,938-image zip;
  4 of those failed face-detection alignment, hence 2,934 aligned images).

## 2. Official KDEF subject IDs (manifest, gender-labeled)

The official release in `C:\Users\b\Downloads\official-kdef-dataset`
(`kdef_manifest_splits.csv`, sessions A/B) names actors `M01–M35` / `F01–F35`
(35 men + 35 women; **not all of them appear in our release**). File names are
`S<g><g><num><EMO><angle>.JPG` (e.g. `BM08DIFL.JPG` = session B, male 08,
disgust, F-center angle).

## 3. Our actor → official-subject map

`kdef_actor_naming_map.csv` (this directory) is the definitive 140-row map:

| Column | Meaning |
|---|---|
| `our_actor_id` | actor 0–139 (key into `aligned/kdef`) |
| `official_subject_id` | e.g. `M08` — filled for 69 actors |
| `gender` | M/F — **all 140 filled** (70 M / 70 F, matching the published KDEF cast split) |
| `match_cosine` | TransFace best-pair cosine from the identity match (69 matched actors only) |
| `source` | `identity_match + visual verification` (69) or `manual_visual_labeling` (71) |
| `official_ref_example` | example official file name for a manual cross-check |

Derivation: 69 of our actors were matched to the manifest via TransFace
identity embeddings (max cosine over up to 8 images per side) and every pair
was visually verified; Kaggle subject `F26` is **not present** in our release.
The remaining 71 actors were labeled by direct visual inspection of frontal
neutral references. The 70 M / 70 F total (published KDEF composition) acts as
a checksum on the labeling.

Related files: `kdef_labels.csv` (main labels, backup `kdef_labels.csv.bak_no_gender`),
`kdef_gender_provenance.csv` (per-actor gender + source),
`kdef_actor_gender_tf.csv` (raw matcher output, superset incl. the excluded F26 row).
