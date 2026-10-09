"""Generate labels for KDEF aligned images (CANONICAL naming).

Filenames in aligned/kdef/ follow convention: <EMOTION>_<actor>_<shot>.jpg
  e.g. ANG_0_25.jpg, HAP_42_3.jpg, SUR_100_1.jpg

- <actor> is our canonical 0–139 actor ID (the key for all labels,
  embeddings and analysis; see datasets/labels/KDEF_provenance_LOCAL.md).
- Gender is joined from datasets/labels/kdef_actor_naming_map.csv
  (author-verified: 140 actors, 70 M / 70 F). If that map is missing or an
  actor is unmapped, the run aborts instead of writing blank genders.
- SAFETY: the script refuses to write if the target folder does not look
  like the canonical subset (>1% unparseable filenames, e.g. when it is
  pointed at the official KDEF release with names like AF21DIHR.jpg), and
  it always backs up an existing kdef_labels.csv before overwriting.

NEVER run this against the official release folder — it would clobber the
canonical subset labels.
"""
import os
import csv
import shutil
import sys
from datetime import datetime
from tqdm import tqdm

headers = [
    'Name', 'Path', 'Identity', 'Gender_code', 'Gender', 'Age',
    'Race_code', 'Race', 'date of birth', 'Emotion_code',
    'Neutral', 'Anger', 'Scream', 'Contempt', 'Disgust', 'Fear',
    'Happy', 'Sadness', 'Surprise',
    'Sun glasses', 'Scarf', 'Eyeglasses', 'Beard', 'Hat', 'Angle'
]

# Emotion mapping from filename prefix to column headers and codes
EMOTION_PREFIX = {
    'ANG': ('Anger', 1),
    'CON': ('Contempt', 3),
    'DIS': ('Disgust', 4),
    'FEA': ('Fear', 5),
    'HAP': ('Happy', 6),
    'NEU': ('Neutral', 0),
    'SAD': ('Sadness', 7),
    'SCR': ('Scream', 2),
    'SUR': ('Surprise', 8),
}


def main():
    # Use absolute path from the script's location to avoid working-dir issues
    script_dir = os.path.dirname(os.path.abspath(__file__))
    toolkit_root = os.path.dirname(script_dir)  # deid-toolkit/
    directory = os.path.join(
        toolkit_root, "root_dir", "datasets", "aligned", "kdef"
    )

    labels_kdef = []
    identities_seen = set()

    img_names = sorted([
        img for img in os.listdir(directory)
        if img.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    print(f"Processing {len(img_names)} images from KDEF...")

    # ── SAFETY PRE-FLIGHT: the folder must contain canonical names ──────
    # (protects against accidentally pointing at the official release,
    # whose filenames like AF21DIHR.jpg would clobber the subset labels)
    bad_names = []
    for img_name in img_names:
        base = os.path.splitext(img_name)[0]
        parts = base.split('_')
        ok = (len(parts) == 3 and parts[0] in EMOTION_PREFIX
              and parts[1].isdigit() and parts[2].isdigit())
        if not ok:
            bad_names.append(img_name)
    if not img_names or (len(bad_names) / len(img_names)) > 0.01:
        print("\nERROR: this folder does NOT look like the canonical KDEF subset "
              "(<EMOTION>_<actor>_<shot>.jpg, actor 0–139).\n"
              f"  {len(bad_names)}/{len(img_names)} filenames unparseable; first few: "
              f"{bad_names[:5]}\n"
              "  Refusing to overwrite kdef_labels.csv. If you are processing the "
              "official release, redirect --output or fix the directory argument."
              )
        sys.exit(1)

    for img_name in tqdm(img_names, desc="KDEF labels"):
        img_path = os.path.join(directory, img_name)

        # Parse filename: <EMOTION>_<SEQ>_<SHOT>.jpg
        base = os.path.splitext(img_name)[0]  # e.g. "ANG_0_25"
        parts = base.split('_')
        if len(parts) != 3:
            print(f"Warning: unexpected filename format: {img_name}, skipping")
            continue

        emotion_prefix, seq_str, shot_str = parts

        # Validate emotion prefix
        if emotion_prefix not in EMOTION_PREFIX:
            print(f"Warning: unknown emotion prefix '{emotion_prefix}' in {img_name}")
            continue

        try:
            seq_id = int(seq_str)
        except ValueError:
            print(f"Warning: invalid subject ID '{seq_str}' in {img_name}, skipping")
            continue

        # Get emotion mapping
        emotion_col, emotion_code = EMOTION_PREFIX[emotion_prefix]

        # Build label row following deid-toolkit conventions.
        # Gender is filled in AFTERWARDS from the verified per-actor map
        # (kdef_actor_naming_map.csv); kept as '' here until the join.
        row = {
            'Name': img_name,
            'Path': 'root_dir/datasets/aligned/kdef/' + img_name,
            'Identity': seq_str,
            'Gender_code': '',
            'Gender': '',
            'Age': '',          # KDEF subjects are young adults (19-34) but ages not recorded
            'Race_code': '',    # All subjects are Caucasian (Swedish) — could be set if needed
            'Race': '',
            'date of birth': '',
            'Emotion_code': emotion_code,
        }

        # Set only the matching emotion column to 1, others stay empty
        for em in ['Neutral', 'Anger', 'Scream', 'Contempt', 'Disgust',
                    'Fear', 'Happy', 'Sadness', 'Surprise']:
            row[em] = 1 if em == emotion_col else ''

        # Metadata columns (all empty — not available for KDEF)
        for extra in ['Sun glasses', 'Scarf', 'Eyeglasses', 'Beard', 'Hat', 'Angle']:
            row[extra] = ''

        labels_kdef.append(row)
        identities_seen.add(seq_str)

    # Write output CSV to the same location as other deid-toolkit labels
    output_path = os.path.join(
        toolkit_root, "root_dir", "datasets", "labels", "kdef_labels.csv"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ── GENDER JOIN: verified per-actor map (author-verified, 70 M / 70 F) ──
    actor_map_path = os.path.join(os.path.dirname(output_path), "kdef_actor_naming_map.csv")
    if not os.path.exists(actor_map_path):
        print("\nERROR: gender map not found: "
              f"{actor_map_path}\nRefusing to write labels with blank genders."
              )
        sys.exit(1)
    actor_gender = {}
    with open(actor_map_path, "r", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            actor_gender[str(r["our_actor_id"]).strip()] = r["gender"].strip()

    GENDER_CODE = {"M": "1", "F": "-1"}
    unmapped = set()
    n_m = n_f = 0
    for row in labels_kdef:
        g = actor_gender.get(row["Identity"])
        if g in ("M", "F"):
            row["Gender"] = g
            row["Gender_code"] = GENDER_CODE[g]
            n_m += g == "M"
            n_f += g == "F"
        else:
            unmapped.add(row["Identity"])
    if unmapped:
        print("\nERROR: actors missing from kdef_actor_naming_map.csv: "
              f"{sorted(unmapped, key=int)[:20]}"
              f" ({len(unmapped)} total). Refusing to write partial-gender labels."
              )
        sys.exit(1)

    # ── BACKUP the existing labels (never destroy without a copy) ────────
    if os.path.exists(output_path):
        backup_path = output_path + ".bak_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy2(output_path, backup_path)
        print(f"\nBacked up existing labels -> {os.path.basename(backup_path)}")

    with open(output_path, "w", newline='') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=headers)
        csv_writer.writeheader()
        csv_writer.writerows(labels_kdef)

    print(f"\nDone! Wrote {len(labels_kdef)} labels to {output_path}")
    print(f"Unique identities: {len(identities_seen)}")
    print(f"Gender: {n_m} M / {n_f} F  (expected 70/70 for the 140-actor KDEF subset)")

    # Print verification sample
    with open(output_path, "r") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)
        count = 0
        for row in reader:
            if len(labels_kdef) <= 1500 or count % 500 == 0:
                print(", ".join(row))
            count += 1


if __name__ == "__main__":
    main()
