import os
import csv
from tqdm import tqdm

headers = ['Name', 'Path', 'Identity', 'Gender_code', 'Gender', 'Age', 'Race_code', 'Race', 'date of birth', 'Emotion_code',
           'Neutral', 'Anger', 'Scream', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise',
           'Sun glasses', 'Scarf', 'Eyeglasses', 'Beard', 'Hat', 'Angle']

directory = os.path.join('root_dir', 'datasets', 'aligned', 'utkface')
output_file = os.path.join('root_dir', 'datasets', 'labels', 'utkface_labels.csv')
labels_utkface = []
race_dict = {'0': 'White', '1': 'Black', '2': 'Asian', '3': 'Indian', '4': 'Other'}

image_files = [img_name for img_name in os.listdir(directory) if img_name.lower().endswith(('.png', '.jpg', '.jpeg'))]
nb_img = len(image_files)

for img_name in tqdm(image_files, desc="Processing Images"):
    img_path = os.path.join(directory, img_name)
    label = {
        'Name': img_name, 'Path': img_path, 'Identity': '', 'Gender_code': '', 'Gender': '', 'Age': '', 'Race_code': '',
        'Race': '', 'date of birth': '', 'Emotion_code': '', 'Neutral': '', 'Anger': '', 'Scream': '',
        'Contempt': '', 'Disgust': '', 'Fear': '', 'Happy': '', 'Sadness': '', 'Surprise': '',
        'Sun glasses': '', 'Scarf': '', 'Eyeglasses': '', 'Beard': '', 'Hat': '', 'Angle': ''
    }

    infos = img_name.split('.')[0].split('_')
    label['Age'] = infos[0]
    if infos[1] == '0':
        label['Gender'] = 'Male'
        label['Gender_code'] = 1
    elif infos[1] == '1':
        label['Gender'] = 'Female'
        label['Gender_code'] = -1
    else:
        print('Error gender:', infos[1])

    if infos[2] in race_dict:
        label['Race'] = race_dict[infos[2]]
    else:
        print('Error race:', infos[2])

    labels_utkface.append(label)

os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Write the labels to the CSV file
with open(output_file, "w", newline='') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=headers)
    csv_writer.writeheader()
    csv_writer.writerows(labels_utkface)

# Read and print some lines from the CSV file to verify
with open(output_file, "r", newline='') as csv_file:
    csv_reader = csv.reader(csv_file)
    for c, line in enumerate(csv_reader, 1):
        if c % 500 == 0:
            print(line)

print("CSV file created:", output_file)
