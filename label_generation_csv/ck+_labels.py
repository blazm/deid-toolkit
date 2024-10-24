import os
import csv
from tqdm import tqdm


headers = ['Name', 'Path', 'Identity', 'Gender_code', 'Gender', 'Age', 'Race_code', 'Race', 'date of birth', 'Emotion_code',
           'Neutral', 'Anger', 'Scream', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise', 'Sun glasses',
           'Scarf', 'Eyeglasses', 'Beard', 'Hat', 'Angle']

directory = os.path.join("root_dir","datasets","aligned","ck+")
emotion_directory = os.path.join("root_dir","datasets","labels","doc","ck+","Emotion_labels")

labels_ck = []

# Dictionary to map emotion labels to codes
emotion_dict = {'Neutral': 0, 'Anger': 1, 'Scream': 2, 'Contempt': 3, 'Disgust': 4,
                'Fear': 5, 'Happy': 6, 'Sadness': 7, 'Surprise': 8}

img_list = [img_name for img_name in os.listdir(directory) if img_name.lower().endswith(('.png', '.jpg', '.jpeg'))]

for img_name in tqdm(img_list, desc="Processing images"):
    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(directory, img_name)
        
        # Initialize a label dictionary with default values
        label = {
            'Name': img_name, 'Path': img_path, 'Identity': '', 'Gender_code': '', 'Gender': '', 'Age': '',
            'Race_code': '', 'Race': '', 'date of birth': '', 'Emotion_code': '', 'Neutral': '', 'Anger': '',
            'Scream': '', 'Contempt': '', 'Disgust': '', 'Fear': '', 'Happy': '', 'Sadness': '', 'Surprise': '',
            'Sun glasses': '', 'Scarf': '', 'Eyeglasses': '', 'Beard': '', 'Hat': '', 'Angle': 0
        }

        infos = img_name.split('_')
        label['Identity'] = infos[0]

        # Generate the path to the emotion label file
        text_file_name = f"{img_name.split('.')[0]}_emotion.txt"
        text_file_path = os.path.join(emotion_directory, text_file_name)

        if os.path.exists(text_file_path):
            with open(text_file_path, "r") as text_file:
                emotion_value = text_file.readline().strip()
                emotion_value_int = int(emotion_value.split('.')[0])
                
                # Map the emotion value to the corresponding emotion and set the label
                if emotion_value_int == 0:
                    label['Neutral'] = 1
                    label['Emotion_code'] = emotion_dict['Neutral']
                elif emotion_value_int == 1:
                    label['Anger'] = 1
                    label['Emotion_code'] = emotion_dict['Anger']
                elif emotion_value_int == 2:
                    label['Contempt'] = 1
                    label['Emotion_code'] = emotion_dict['Contempt']
                elif emotion_value_int == 3:
                    label['Disgust'] = 1
                    label['Emotion_code'] = emotion_dict['Disgust']
                elif emotion_value_int == 4:
                    label['Fear'] = 1
                    label['Emotion_code'] = emotion_dict['Fear']
                elif emotion_value_int == 5:
                    label['Happy'] = 1
                    label['Emotion_code'] = emotion_dict['Happy']
                elif emotion_value_int == 6:
                    label['Sadness'] = 1
                    label['Emotion_code'] = emotion_dict['Sadness']
                elif emotion_value_int == 7:
                    label['Surprise'] = 1
                    label['Emotion_code'] = emotion_dict['Surprise']
        
        labels_ck.append(label)

output_path = os.path.join("root_dir","datasets","labels","ck+_labels.csv")
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w", newline='') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=headers)
    csv_writer.writeheader()
    csv_writer.writerows(labels_ck)

# Read and print some lines from the CSV file to verify
with open(output_path, "r", newline='') as csv_file:
    csv_reader = csv.reader(csv_file)
    c = 0
    for line in csv_reader:
        c += 1
        if c % 500 == 0:
            print(line)
