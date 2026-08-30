"""Generate labels for RAF-DB aligned images.

Filenames follow convention: Rafd<view>_<subject>_<ethnicity>_<gender>_<expression>_frontal.jpg
  e.g. Rafd090_45_Caucasian_female_angry_frontal.jpg

Emotion codes match the deid-toolkit standard emotion dictionary.
Contemptuous images were already removed from aligned/rafd/.
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

# Mapping from filename expression word -> (CSV column, emotion code)
# Matches deid-toolkit standard: Neutral=0, Anger=1, Scream=2, Contempt=3,
# Disgust=4, Fear=5, Happy=6, Sadness=7, Surprise=8
EXPRESSION_MAP = {
    'angry': ('Anger', 1),
    'disgusted': ('Disgust', 4),
    'fearful': ('Fear', 5),
    'happy': ('Happy', 6),
    'neutral': ('Neutral', 0),
    'sad': ('Sadness', 7),
    'surprised': ('Surprise', 8),
}

def main():
    # Determine paths from script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    toolkit_root = os.path.dirname(script_dir)
    aligned_dir = os.path.join(toolkit_root, "root_dir", "datasets", "aligned", "rafd")

    img_names = sorted([
        n for n in os.listdir(aligned_dir)
        if n.lower().endswith(('.jpg', '.jpeg', '.png')) and 'frontal' in n
    ])

    print(f"Processing {len(img_names)} RAF-DB images...")

    labels = []
    identities_seen = set()

    for img_name in tqdm(img_names, desc="RAF labels"):
        # Parse: Rafd<view>_<subject>_<ethnicity>_<gender>_<expression>_frontal.jpg
        base = os.path.splitext(img_name)[0]  # e.g. "Rafd090_45_Caucasian_female_angry_frontal"
        parts = base.split('_')

        if len(parts) < 6:
            print(f"Warning: unexpected format: {img_name}")
            continue

        view_str = parts[0]    # e.g. "Rafd090" -> angle 90
        subject_id = parts[1]  # e.g. "45"
        ethnicity = parts[2]   # e.g. "Caucasian", "Black", "Indian"
        gender_str = parts[3]  # "female" or "male"
        expr_word = parts[4]   # e.g. "angry"

        if expr_word not in EXPRESSION_MAP:
            print(f"Warning: unknown expression '{expr_word}' in {img_name}")
            continue

        gender_col, emotion_code = EXPRESSION_MAP[expr_word]

        # Gender code: -1=Female, 1=Male (matches arface convention)
        gender_code = -1 if gender_str == 'female' else 1
        gender_label = gender_str.capitalize()

        row = {
            'Name': img_name,
            'Path': f'root_dir/datasets/aligned/rafd/{img_name}',
            'Identity': subject_id,
            'Gender_code': gender_code,
            'Gender': gender_label,
            'Age': '',               # RAF-DB does not provide age per-subject
            'Race_code': '',         # Could map ethnicity if desired
            'Race': ethnicity,       # ethnicity from filename
            'date of birth': '',
            'Emotion_code': emotion_code,
        }

        # Set only the matching emotion column to 1
        for em in ['Neutral', 'Anger', 'Scream', 'Contempt', 'Disgust',
                    'Fear', 'Happy', 'Sadness', 'Surprise']:
            row[em] = 1 if em == gender_col else ''

        # Metadata columns (all empty)
        for extra in ['Sun glasses', 'Scarf', 'Eyeglasses', 'Beard', 'Hat', 'Angle']:
            row[extra] = ''

        labels.append(row)
        identities_seen.add(subject_id)

    output_path = os.path.join(
        toolkit_root, "root_dir", "datasets", "labels", "rafd-frontal_aligned_labels.csv"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(labels)

    print(f"\nDone! Wrote {len(labels)} labels to {output_path}")
    print(f"Unique identities (subjects): {len(identities_seen)}")

    # Print verification sample
    with open(output_path, "r") as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # skip header
        count = 0
        for row in reader:
            if len(labels) <= 1500 or count % 500 == 0:
                print(", ".join(row))
            count += 1


if __name__ == "__main__":
    main()
