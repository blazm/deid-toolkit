import os
import csv
from tqdm import tqdm

headers = ['Name', 'Path', 'Identity', 'Gender_code', 'Gender', 'Age', 'Race_code', 'Race', 'date of birth',
           'Emotion_code', 'Neutral', 'Anger', 'Scream', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sadness',
           'Surprise', 'Sun glasses', 'Scarf', 'Eyeglasses', 'Beard', 'Hat', 'Angle']

base_dir = os.path.join('root_dir', 'datasets', 'original', 'Arc2Face_data', 'img')
output_directory = os.path.join('root_dir', 'datasets', 'labels')
output_path = os.path.join(output_directory, "Arc2Face_labels.csv")

labels = []

for person_id in tqdm(sorted(os.listdir(base_dir)), desc="Processing Arc2Face persons"):
    person_dir = os.path.join(base_dir, person_id)
    if not os.path.isdir(person_dir):
        continue
    for img_name in os.listdir(person_dir):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        rel_path = os.path.join('root_dir', 'datasets', 'original', 'Arc2Face_data', 'img', person_id, img_name).replace('\\', '/')
        labels.append({
            'Name': img_name, 'Path': rel_path, 'Identity': person_id,
            'Gender_code': '', 'Gender': '', 'Age': '', 'Race_code': '', 'Race': '',
            'date of birth': '', 'Emotion_code': '', 'Neutral': '', 'Anger': '',
            'Scream': '', 'Contempt': '', 'Disgust': '', 'Fear': '', 'Happy': '',
            'Sadness': '', 'Surprise': '', 'Sun glasses': '', 'Scarf': '', 'Eyeglasses': '',
            'Beard': '', 'Hat': '', 'Angle': ''
        })

os.makedirs(output_directory, exist_ok=True)

with open(output_path, "w", newline='') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=headers)
    csv_writer.writeheader()
    csv_writer.writerows(labels)

print(f"Wrote {len(labels)} rows to {output_path}")
print(f"Unique identities: {len(set(l['Identity'] for l in labels))}")
