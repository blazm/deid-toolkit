import os
import csv
import numpy as np
from tqdm import tqdm

headers = [
    "Name",
    "Path",
    "Identity",
    "Gender_code",
    "Gender",
    "Age",
    'Race_code',
    "Race",
    "date of birth",
    "Emotion_code",
    "Neutral",
    "Anger",
    "Scream",
    "Contempt",
    "Disgust",
    "Fear",
    "Happy",
    "Sadness",
    "Surprise",
    "Sun glasses",
    "Scarf",
    "Eyeglasses",
    "Beard",
    "Hat",
    "Angle",
]

directory = os.path.join('root_dir', 'datasets', 'aligned', 'morph')
csv_file_out = os.path.join('root_dir', 'datasets', 'labels', 'morph_labels.csv')
all_data = os.path.join('root_dir', 'datasets', 'labels', 'doc', 'morph', 'all_data.csv')
labels_morph = []

image_files = [img for img in os.listdir(directory) if img.lower().endswith((".png", ".jpg", ".jpeg"))]

for img_name in tqdm(image_files, desc="Processing Images"):
    img_path = os.path.join(directory, img_name)
    infos = img_name.split(".")[0]
    id = infos.split("_")[0]
    label = {
        "Name": img_name,
        "Path": img_path,
        "Identity": id,
        "Gender_code": "",
        "Gender": "",
        "Age": "",
        'Race_code': '',
        "Race": "",
        "date of birth": "",
        "Emotion_code": "",
        "Neutral": "",
        "Anger": "",
        "Scream": "",
        "Contempt": "",
        "Disgust": "",
        "Fear": "",
        "Happy": "",
        "Sadness": "",
        "Surprise": "",
        "Sun glasses": "",
        "Scarf": "",
        "Eyeglasses": "",
        "Beard": "",
        "Hat": "",
        "Angle": "",
    }

    if len(infos.split('_')[1]) == 4:
        gender = infos.split('_')[1][1]
        age = infos.split('_')[1][2:]
    else:
        gender = infos.split('_')[1][2]
        age = infos.split('_')[1][3:]

    if len(age) != 2:
        print("Error age:", age)

    if gender == 'M':
        gender = 'Male'
        gender_code = 1
    elif gender == 'F':
        gender = 'Female'
        gender_code = -1
    else:
        print('Error gender:', gender)

    label['Gender'] = gender
    label['Gender_code'] = gender_code
    label['Age'] = age

    if id[:2] == "00":
        id_to_search = id[2:]
    elif id[0] == "0":
        id_to_search = id[1:]
    else:
        id_to_search = id

    with open(all_data, "r", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames

        if fieldnames[0].startswith('\ufeff'):
            fieldnames[0] = fieldnames[0][1:]

        for row in reader:
            if row['id_num'] == id_to_search:
                race = row["race"]
                date_of_birth = row["dob"]
                facial_hair = row["facial_hair"]
                glasses = row["glasses"]

                race_mapping = {
                    "W": "White",
                    "White": "White",
                    "B": "Black",
                    "African-American.Black": "Black",
                    "I": "Indian",
                    "H": "Hispanic",
                    "O": "Other",
                    "other": "Other",
                    "A": "Asian"
                }

                label["Race"] = race_mapping.get(race, f"Error race: {race}")

                if len(date_of_birth.split("-")[0]) == 4:
                    date_parts = date_of_birth.split("-")
                    date_of_birth = f"{date_parts[1]}.{date_parts[2]}.{date_parts[0]}"
                elif date_of_birth[2] == '/':
                    date_parts = date_of_birth.split('/')
                    date_of_birth = f"{date_parts[1]}.{date_parts[0]}.{date_parts[2]}"
                label['date of birth'] = date_of_birth

                if glasses == "1":
                    label["Eyeglasses"] = 1
                elif glasses=='0':
                    label["Eyeglasses"] = -1

                if facial_hair == "1":
                    label["Beard"] = 1 
                elif facial_hair == "0":
                    label["Beard"] = -1

    labels_morph.append(label)

with open(csv_file_out, "w", newline="") as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=headers)
    csv_writer.writeheader()
    csv_writer.writerows(labels_morph)

with open(csv_file_out, "r", newline="") as csv_file:
    csv_reader = csv.reader(csv_file)
    for c, line in enumerate(csv_reader):
        if c % 500 == 0:
            print(line)
