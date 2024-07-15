import os
import csv
from tqdm import tqdm

headers = ['Name', 'Path', 'Identity', 'Gender_code', 'Gender', 'Age', 'Race_code', 'Race', 'date of birth', 'Emotion_code',
           'Neutral', 'Anger', 'Scream', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise',
           'Sun glasses', 'Scarf', 'Eyeglasses', 'Beard', 'Hat', 'Angle']

directory = os.path.join('root_dir', 'datasets', 'aligned', 'rafd')
labels_rafd = []

pose_dict = {'000': 90, '045': 45, '090': 0, '135': -45, '180': -90}
emotion_dict = {'Neutral': 0, 'Anger': 1, 'Scream': 2, 'Contempt': 3, 'Disgust': 4,
                'Fear': 5, 'Happy': 6, 'Sadness': 7, 'Surprise': 8}

image_files = [img_name for img_name in os.listdir(directory) if img_name.lower().endswith(('.png', '.jpg', '.jpeg'))]

for img_name in tqdm(image_files, desc="Processing Images"):
    img_path = os.path.join(directory, img_name)
    
    infos = img_name.split('.')[0]
    infos = infos.split('_')

    id = infos[1]
    label = {
        'Name': img_name, 'Path': img_path, 'Identity': id, 'Gender_code': '', 'Gender': '',
        'Age': '', 'Race_code': '', 'Race': '', 'date of birth': '', 'Emotion_code': '',
        'Neutral': '', 'Anger': '', 'Scream': '', 'Contempt': '', 'Disgust': '',
        'Fear': '', 'Happy': '', 'Sadness': '', 'Surprise': '', 'Sun glasses': '',
        'Scarf': '', 'Eyeglasses': '', 'Beard': '', 'Hat': '', 'Angle': ''
    }

    angle = infos[0][4:7]
    for i in pose_dict:
        if i == angle:
            label['Angle'] = pose_dict[angle]

    race = infos[2]
    label['Race'] = race

    gender = infos[3]
    if gender == 'female':
        label['Gender'] = 'Female'
        label['Gender_code'] = -1
    elif gender == 'male':
        label['Gender'] = 'Male'
        label['Gender_code'] = 1
    else:
        print("error gender", gender)

    emotion = infos[4]
    if emotion == 'angry':
        label['Anger'] = 1
        label['Emotion_code'] = emotion_dict['Anger']
    elif emotion == 'contemptuous':
        label['Contempt'] = 1
        label['Emotion_code'] = emotion_dict['Contempt']
    elif emotion == 'disgusted':
        label['Disgust'] = 1
        label['Emotion_code'] = emotion_dict['Disgust']
    elif emotion == 'fearful':
        label['Fear'] = 1
        label['Emotion_code'] = emotion_dict['Fear']
    elif emotion == 'happy':
        label['Happy'] = 1
        label['Emotion_code'] = emotion_dict['Happy']
    elif emotion == 'neutral':
        label['Neutral'] = 1
        label['Emotion_code'] = emotion_dict['Neutral']
    elif emotion == 'sad':
        label['Sadness'] = 1
        label['Emotion_code'] = emotion_dict['Sadness']
    elif emotion == 'surprised':
        label['Surprise'] = 1
        label['Emotion_code'] = emotion_dict['Surprise']
    else:
        print('error emotion', emotion)
    
    labels_rafd.append(label)

output_path = os.path.join('root_dir', 'datasets', 'labels', 'rafd_labels.csv')
with open(output_path, "w", newline='') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=headers)
    csv_writer.writeheader()
    csv_writer.writerows(labels_rafd)

print("CSV file created:", output_path)
