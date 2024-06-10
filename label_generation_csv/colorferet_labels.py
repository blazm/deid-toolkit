import os
import csv
from tqdm import tqdm

headers = ['Name', 'Path', 'Identity', 'Gender_code', 'Gender', 'Age', 'Race_code', 'Race', 'date of birth', 'Emotion_code',
           'Neutral', 'Anger', 'Scream', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise',
           'Sun glasses', 'Scarf', 'Eyeglasses', 'Beard', 'Hat', 'Angle']

directory = "root_dir/datasets/aligned/colorferet"
ground_truths_directory = "root_dir/datasets/labels/doc/colorferet/ground_truths/name_value"

labels_colorferet = []

# Define a dictionary to map pose names to angles
pose_angle_dict = {'fa': '0', 'fb': '0', 'pl': '90', 'hl': '67.5', 'ql': '22.5', 'pr': '-90',
                   'hr': '-67.5', 'qr': '-22.5', 'ra': '45', 'rb': '15', 'rc': '-15', 'rd': '-45', 're': '-75'}

img_names = [img for img in os.listdir(directory) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

for img_name in tqdm(img_names, desc="Processing images"):
    img_path = os.path.join(directory, img_name)
    
    infos = img_name.split('_')
    id = infos[0]
    label = {
        'Name': '', 'Path': '', 'Identity': '', 'Gender_code': '', 'Gender': '', 'Age': '', 'Race_code': '',
        'Race': '', 'date of birth': '', 'Emotion_code': '', 'Neutral': '', 'Anger': '', 'Scream': '',
        'Contempt': '', 'Disgust': '', 'Fear': '', 'Happy': '', 'Sadness': '', 'Surprise': '', 
        'Sun glasses': '', 'Scarf': '', 'Eyeglasses': '', 'Beard': '', 'Hat': '', 'Angle': ''
    }

    # Handle optional flags for pose and eyeglasses
    if len(infos) == 4:
        optional_flag = infos[3].split('.')[0]
        pose_name = infos[2]
        label['Eyeglasses'] = 1 if optional_flag in ['a', 'c'] else -1
    else:
        pose_name = infos[2].split('.')[0]
    
    # Extract capture date and calculate age
    date_capture = infos[1]  # YYMMDD
    capture_year = int('19' + date_capture[0:2])
    text_file_path = os.path.join(ground_truths_directory, id + '.txt')
    with open(text_file_path, 'r') as file:  # Find the year of birth in the .txt label
        lines = file.readlines()
    year_of_birth = int(lines[2].split('=')[1])
    age = capture_year - year_of_birth 

    # Extract race from the .txt label
    race = lines[3].split('=')[1].strip()  # Remove trailing whitespace
    if race == 'Black-or-African-American':
        race = 'Black'

    # Extract gender from the .txt label
    gender = lines[1].split('=')[1].strip()  # Remove trailing whitespace
    if gender.startswith('Male'):
        gender_code = 1
        gender = 'Male'
    else:
        gender_code = -1
        gender = 'Female'
    
    label['Angle'] = pose_angle_dict.get(pose_name, '')
    
    label.update({
        'Name': img_name, 'Path': img_path, 'Identity': id, 'Gender_code': gender_code, 'Gender': gender, 
        'Age': age, 'Race_code': '', 'Race': race, 'date of birth': year_of_birth,
        'Emotion_code': '', 'Neutral': '', 'Anger': '', 'Scream': '', 'Contempt': '', 'Disgust': '',
        'Fear': '', 'Happy': '', 'Sadness': '', 'Surprise': '', 'Sun glasses': '',
        'Scarf': '', 'Eyeglasses': label['Eyeglasses'], 'Beard': '', 'Hat': '', 'Angle': label['Angle']
    })
    labels_colorferet.append(label)

output_path = "root_dir/datasets/labels/colorferet_labels.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w", newline='') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=headers)
    csv_writer.writeheader()
    csv_writer.writerows(labels_colorferet)

# Read and print some lines from the CSV file to verify
with open(output_path, "r", newline='') as csv_file:
    csv_reader = csv.reader(csv_file)
    c = 0
    for line in csv_reader:
        c += 1
        if c % 500 == 0:
            print(line)
