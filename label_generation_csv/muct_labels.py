import os
import csv
import numpy as np
import time

headers = ['Name', 'Path', 'Identity', 'Gender_code', 'Gender', 'Age', 'Race_code', 'Race', 'date of birth',
           'Emotion_code', 'Neutral', 'Anger', 'Scream', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sadness',
           'Surprise', 'Sun glasses', 'Scarf', 'Eyeglasses', 'Beard', 'Hat', 'Angle']

directory = "root_dir/datasets/original/muct/img"
output_directory = "root_dir/datasets/labels"
output_path = os.path.join(output_directory, "muct_labels.csv")

labels_muct = []
nb_img = len(os.listdir(directory))
time_start = time.time()

for c, img_name in enumerate(os.listdir(directory), 1):
    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
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
        
        if glasses == 'g':
            label['Eyeglasses'] = 1
        else:
            label['Eyeglasses'] = -1

        labels_muct.append(label)

    if c == 1:
        time_per_image = time.time() - time_start
        estimated_total_duration_sec = nb_img * time_per_image
        estimated_total_duration_hours = estimated_total_duration_sec / 3600
        print(f"Estimated duration: {estimated_total_duration_hours:.2f} hours (~ {estimated_total_duration_sec:.2f} sec)")

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Write the labels to the CSV file
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