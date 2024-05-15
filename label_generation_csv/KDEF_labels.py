import os
import csv
import numpy as np

headers = ['Name', 'Path', 'Identity', 'Gender_code', 'Gender', 'Age', 'Race_code', 'Race', 'date of birth', 
           'Emotion_code', 'Neutral', 'Anger', 'Scream', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sadness', 
           'Surprise', 'Sun glasses', 'Scarf', 'Eyeglasses', 'Beard', 'Hat', 'Angle']

directory = "root_dir/datasets/original/kdef/img"

labels_kdef = []

pose_angle_dict = {'S': '0', 'FL': '90', 'HL': '67.5', 'FR': '-90', 'HR': '-67.5'}
emotion_dict = {'Neutral': 0, 'Anger': 1, 'Scream': 2, 'Contempt': 3, 'Disgust': 4,
                'Fear': 5, 'Happy': 6, 'Sadness': 7, 'Surprise': 8}

nb_img = len(os.listdir(directory))

for c, img_name in enumerate(os.listdir(directory), 1):
    progression = np.round(100 * c / nb_img, 2)
    print(f"\n Progression: {progression}% \n") 

    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(directory, img_name)
        
        infos = img_name.split('.')[0]
        gender = infos[1]
        id = infos[2:4]
        emotion = infos[4:6]
        angle = infos[6:]

        label = {
            'Name': img_name, 'Path': img_path, 'Identity': id, 'Gender_code': '', 'Gender': '', 
            'Age': '', 'Race_code': '', 'Race': '', 'date of birth': '', 'Emotion_code': '', 'Neutral': '', 
            'Anger': '', 'Scream': '', 'Contempt': '', 'Disgust': '', 'Fear': '', 'Happy': '', 'Sadness': '', 
            'Surprise': '', 'Sun glasses': '', 'Scarf': '', 'Eyeglasses': '', 'Beard': '', 'Hat': '', 'Angle': ''
        }

        if gender == 'F':
            label['Gender_code'] = -1
            label['Gender'] = 'Female'
        else:
            label['Gender_code'] = 1
            label['Gender'] = 'Male'
        
        if angle in pose_angle_dict:
            label['Angle'] = pose_angle_dict[angle]

        if emotion == 'AF':
            label['Fear'] = 1
            label['Emotion_code'] = emotion_dict['Fear']
        elif emotion == 'AN':
            label['Anger'] = 1
            label['Emotion_code'] = emotion_dict['Anger']
        elif emotion == 'DI':
            label['Disgust'] = 1
            label['Emotion_code'] = emotion_dict['Disgust']
        elif emotion == 'HA':
            label['Happy'] = 1
            label['Emotion_code'] = emotion_dict['Happy']
        elif emotion == 'NE':
            label['Neutral'] = 1
            label['Emotion_code'] = emotion_dict['Neutral']
        elif emotion == 'SA':
            label['Sadness'] = 1
            label['Emotion_code'] = emotion_dict['Sadness']
        elif emotion == 'SU':
            label['Surprise'] = 1
            label['Emotion_code'] = emotion_dict['Surprise']

        labels_kdef.append(label)

# Ensure the output directory exists
output_directory = "root_dir/datasets/labels"
output_path = os.path.join(output_directory, "kdef_labels.csv")
os.makedirs(output_directory, exist_ok=True)

# Write the labels to the CSV file
with open(output_path, "w", newline='') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=headers)
    csv_writer.writeheader()
    csv_writer.writerows(labels_kdef)

# Read and print some lines from the CSV file to verify
with open(output_path, "r", newline='') as csv_file:
    csv_reader = csv.reader(csv_file)
    for c, line in enumerate(csv_reader, 1):
        if c % 500 == 0:
            print(line)