
#### MOST OF THIS CSV FILE HAVE BEEN HAND WRITTEN
import os
import csv
import numpy as np

headers = ['Name', 'Path', 'Identity', 'Gender_code', 'Gender', 'Age', 'Race', 'date of birth', 'Emotion_code', 
           'Neutral', 'Anger', 'Scream', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise', 
           'Sun glasses', 'Scarf', 'Eyeglasses', 'Beard', 'Hat', 'Angle']

directory = os.path.join('root_dir', 'datasets', 'aligned', 'fri')
labels_fri = []

nb_img = len(os.listdir(directory))

for c, img_name in enumerate(os.listdir(directory), 1):
    progression = np.round(100 * c / nb_img, 2)
    print(f"\n Progression: {progression}% \n")

    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(directory, img_name)
        
        label = {
            'Name': img_name, 'Path': img_path, 'Identity': img_name.split('.')[0], 'Gender_code': '', 'Gender': '', 
            'Age': '', 'Race': '', 'date of birth': '', 'Emotion_code': '', 'Neutral': '', 'Anger': '', 
            'Scream': '', 'Contempt': '', 'Disgust': '', 'Fear': '', 'Happy': '', 'Sadness': '', 'Surprise': '', 
            'Sun glasses': '', 'Scarf': '', 'Eyeglasses': '', 'Beard': '', 'Hat': '', 'Angle': 0
        }
        
        labels_fri.append(label)

# Ensure the output directory exists
output_directory = 'root_dir', 'datasets', 'labels'
output_path = os.path.join(output_directory, "fri_labels.csv")
os.makedirs(output_directory, exist_ok=True)

# Write the labels to the CSV file
with open(output_path, "w", newline='') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=headers)
    csv_writer.writeheader()
    csv_writer.writerows(labels_fri)

# Read and print some lines from the CSV file to verify
with open(output_path, "r", newline='') as csv_file:
    csv_reader = csv.reader(csv_file)
    for c, line in enumerate(csv_reader, 1):
        print(line)
        if c % 500 == 0:
            print(line)



