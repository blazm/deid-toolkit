import os
import csv
from tqdm import tqdm

headers = ['Name', 'Path', 'Identity', 'Gender_code', 'Gender', 'Age', 'Race_code', 'Race', 'date of birth',
           'Emotion_code', 'Neutral', 'Anger', 'Scream', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sadness',
           'Surprise', 'Sun glasses', 'Scarf', 'Eyeglasses', 'Beard', 'Hat', 'Angle']

directory = os.path.join('root_dir', 'datasets', 'aligned', 'muct')
output_directory = os.path.join('root_dir', 'datasets', 'labels')
output_path = os.path.join(output_directory, "muct_labels.csv")

labels_muct = []
image_files = [img for img in os.listdir(directory) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
nb_img = len(image_files)

for img_name in tqdm(image_files, desc="Processing Images"):
    img_path = os.path.join(directory, img_name)

    infos = img_name.split('.')[0]
    id = infos[1:4]
    gender = infos[-2]
    glasses = infos[-1]

    label = {
        'Name': img_name, 'Path': img_path, 'Identity': id, 'Gender_code': '', 'Gender': '',
        'Age': '', 'Race_code': '', 'Race': '', 'date of birth': '', 'Emotion_code': '', 'Neutral': '',
        'Anger': '', 'Scream': '', 'Contempt': '', 'Disgust': '', 'Fear': '',
        'Happy': '', 'Sadness': '', 'Surprise': '', 'Sun glasses': '', 'Scarf': '',
        'Eyeglasses': '', 'Beard': '', 'Hat': '', 'Angle': ''
    }

    if gender == 'm':
        label['Gender_code'] = 1
        label['Gender'] = 'Male'
    else:
        label['Gender_code'] = -1
        label['Gender'] = 'Female'

    label['Eyeglasses'] = 1 if glasses == 'g' else -1

    labels_muct.append(label)

os.makedirs(output_directory, exist_ok=True)

with open(output_path, "w", newline='') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=headers)
    csv_writer.writeheader()
    csv_writer.writerows(labels_muct)

# Read and print some lines from the CSV file to verify
with open(output_path, "r", newline='') as csv_file:
    csv_reader = csv.reader(csv_file)
    for c, line in enumerate(csv_reader, 1):
        if c % 500 == 0:
            print(line)
