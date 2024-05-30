import os
import csv
import numpy as np

headers=['Name','Path','Identity','Gender_code','Gender','Age','Race_code','Race','date of birth','Emotion_code',
         'Neutral','Anger','Scream','Contempt','Disgust','Fear','Happy','Sadness','Surprise',
         'Sun glasses','Scarf','Eyeglasses','Beard','Hat','Angle']

directory = "root_dir/datasets/aligned/xm2vts"
labels_xm2vts= []

c=0
for img_name in os.listdir(directory):

    c+=1
    nb_img = len(os.listdir(directory))
    progression = np.round(100*c/nb_img,2)
    print(f"\n Progession: {progression}% \n") 

    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(directory, img_name)
        id=img_name.split('_')[0]
        label = {
        'Name':img_name,'Path':img_path,'Identity': id,'Gender_code':'','Gender': '', 'Age': '','Race_code':'',
        'Race': '', 'date of birth': '','Emotion_code':'','Neutral': '', 'Anger': '', 'Scream': '',
        'Contempt': '', 'Disgust': '','Fear': '', 'Happy': '', 'Sadness': '', 'Surprise': '', 
        'Sun glasses': '','Scarf': '', 'Eyeglasses': '', 'Beard': '', 'Hat': '', 'Angle': ''
    }
    labels_xm2vts.append(label)


with open("root_dir/datasets/labels/xm2vts_labels.csv","w",newline = '') as csv_file:
    csv_writer = csv.DictWriter(csv_file,fieldnames=headers)
    csv_writer.writeheader()
    csv_writer.writerows(labels_xm2vts)


with open("root_dir/datasets/labels/xm2vts_labels.csv","r",newline = '') as csv_file:
    csv_reader = csv.reader(csv_file)
    c=0
    for line in csv_reader:
        c+=1
        if c%500==0:
            print(line)