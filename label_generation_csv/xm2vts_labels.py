import os
import csv
from tqdm import tqdm

headers = ['Name', 'Path', 'Identity', 'Gender_code', 'Gender', 'Age', 'Race_code', 'Race', 'date of birth', 'Emotion_code',
           'Neutral', 'Anger', 'Scream', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise',
           'Sun glasses', 'Scarf', 'Eyeglasses', 'Beard', 'Hat', 'Angle']

directory = "root_dir/datasets/aligned/xm2vts"
output_file = "root_dir/datasets/labels/xm2vts_labels.csv"
labels_xm2vts = []

image_files = [img_name for img_name in os.listdir(directory) if img_name.lower().endswith(('.png', '.jpg', '.jpeg'))]

for img_name in tqdm(image_files, desc="Processing Images"):
    img_path = os.path.join(directory, img_name)
    id = img_name.split('_')[0]
    label = {
        'Name': img_name, 'Path': img_path, 'Identity': id, 'Gender_code': '', 'Gender': '', 'Age': '', 'Race_code': '',
        'Race': '', 'date of birth': '', 'Emotion_code': '', 'Neutral': '', 'Anger': '', 'Scream': '',
        'Contempt': '', 'Disgust': '', 'Fear': '', 'Happy': '', 'Sadness': '', 'Surprise': '',
        'Sun glasses': '', 'Scarf': '', 'Eyeglasses': '', 'Beard': '', 'Hat': '', 'Angle': ''
    }
    labels_xm2vts.append(label)

os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Write the labels to the CSV file
with open(output_file, "w", newline='') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=headers)
    csv_writer.writeheader()
    csv_writer.writerows(labels_xm2vts)

# Read and print some lines from the CSV file to verify
with open(output_file, "r", newline='') as csv_file:
    csv_reader = csv.reader(csv_file)
    for c, line in enumerate(csv_reader, 1):
        if c % 500 == 0:
            print(line)

print("CSV file created:", output_file)
