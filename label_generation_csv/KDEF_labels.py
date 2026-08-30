"""Generate labels for KDEF aligned images.

Filenames in aligned/kdef/ follow convention: <EMOTION>_<SEQ>_<SHOT>.jpg
  e.g. ANG_0_25.jpg, HAP_42_3.jpg, SUR_100_1.jpg

Emotion codes match the existing emotion dictionary used throughout
deid-toolkit for label consistency. Gender is NOT encoded (KDEF
original filenames do not carry gender information and no official
subject-to-gender mapping file was included with the dataset we have).
"""
import os
import csv
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
    'DIS': ('Disgust', 4),
    'FEA': ('Fear', 5),
    'HAP': ('Happy', 6),
    'NEU': ('Neutral', 0),
    'SAD': ('Sadness', 7),
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
        # Gender is NOT known from filenames — left blank (KDEF originals
        # used subject codes like S001 with no gender field in the image names).
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

    with open(output_path, "w", newline='') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=headers)
        csv_writer.writeheader()
        csv_writer.writerows(labels_kdef)

    print(f"\nDone! Wrote {len(labels_kdef)} labels to {output_path}")
    print(f"Unique identities: {len(identities_seen)}")

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
