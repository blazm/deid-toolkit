import os
import csv
from tqdm import tqdm

headers = ['Name','Path','Identity','Gender_code','Gender','Age','Race_code','Race','date of birth','Emotion_code',
           'Neutral','Anger','Scream','Contempt','Disgust','Fear','Happy','Sadness','Surprise',
           'Sun glasses','Scarf','Eyeglasses','Beard','Hat','Angle']

directory = "root_dir/datasets/aligned/arface"

labels_arface = []

emotion_dict = {'Neutral':0,'Anger':1,'Scream':2,'Contempt':3,'Disgust':4,
                'Fear':5,'Happy':6,'Sadness':7,'Surprise':8}


img_list = [img_name for img_name in os.listdir(directory) if img_name.lower().endswith(('.png', '.jpg', '.jpeg'))]
for img_name in tqdm(img_list, desc="Processing images"):
    img_path = os.path.join(directory, img_name)
    infos = img_name.split('-')

    label = {
        'Name': img_name, 'Path': img_path, 'Identity': '', 'Gender_code': '', 'Gender': '', 'Age': '',
        'Race_code': '', 'Race': '', 'date of birth': '', 'Emotion_code': '', 'Neutral': '', 'Anger': '',
        'Scream': '', 'Contempt': '', 'Disgust': '', 'Fear': '', 'Happy': '', 'Sadness': '',
        'Surprise': '', 'Sun glasses': '', 'Scarf': '', 'Eyeglasses': '', 'Beard': '', 'Hat': '',
        'Angle': 0
    }

    label['Identity'] = infos[1]
    gender = 'Female'
    gender_code = -1
    if infos[0] == 'M' or infos[0] == 'm':
        gender = 'Male'
        gender_code = 1
    label['Gender'] = gender
    label['Gender_code'] = gender_code

    attribute_code = int(infos[-1].split('.')[0])  # remove the extension of the name
    if attribute_code in [1, 14]:
        label['Neutral'] = 1
        label['Emotion_code'] = emotion_dict['Neutral']
    elif attribute_code in [2, 15]:
        label['Happy'] = 1
        label['Emotion_code'] = emotion_dict['Happy']
    elif attribute_code in [3, 16]:
        label['Anger'] = 1
        label['Emotion_code'] = emotion_dict['Anger']
    elif attribute_code in [4, 17]:
        label['Scream'] = 1
        label['Emotion_code'] = emotion_dict['Scream']
    elif attribute_code in [8, 9, 10, 21, 22, 23]:
        label['Sun glasses'] = 1
    elif attribute_code in [11, 12, 13, 24, 25, 26]:
        label['Scarf'] = 1
    labels_arface.append(label)

output_path = r"root_dir/datasets/labels/arface_labels.csv"

with open(output_path, "w", newline='') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=headers)
    csv_writer.writeheader()
    csv_writer.writerows(labels_arface)

with open(output_path, "r", newline='') as csv_file:
    csv_reader = csv.reader(csv_file)
    c = 0
    for line in csv_reader:
        c += 1
        if c % 500 == 0:
            print(line)