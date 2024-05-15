#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import csv
import numpy as np

headers=['Name','Path','Identity','Gender_code','Gender','Age','Race_code','Race','date of birth','Emotion_code',
         'Neutral','Anger','Scream','Contempt','Disgust','Fear','Happy','Sadness','Surprise',
         'Sun glasses','Scarf','Eyeglasses','Beard','Hat','Angle']

directory = "root_dir/datasets/original//utkface/img"
labels_utkface= []
race_dict ={'0':'White','1':'Black','2':'Asian','3':'Indian','4':'Other'}

c=0
for img_name in os.listdir(directory):

    c+=1
    nb_img = len(os.listdir(directory))
    progression = np.round(100*c/nb_img,2)
    print(f"\n Progession: {progression}% \n") 

    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(directory, img_name)
        label = {
        'Name':img_name,'Path':img_path,'Identity': '','Gender_code':'','Gender': '', 'Age': '','Race_code':'',
        'Race': '', 'date of birth': '','Emotion_code':'','Neutral': '', 'Anger': '', 'Scream': '',
        'Contempt': '', 'Disgust': '','Fear': '', 'Happy': '', 'Sadness': '', 'Surprise': '', 
        'Sun glasses': '','Scarf': '', 'Eyeglasses': '', 'Beard': '', 'Hat': '', 'Angle': ''
    } 
        
        infos = img_name.split('.')[0]
        infos = infos.split('_')
        label['Age']=infos[0]
        if infos[1]== '0':
            label['Gender']='Male'
            label['Gender_code'] = 1
        elif infos[1]=='1':
            label['Gender']='Female'
            label['Gender_code'] = -1
        else:
            print('error gender:',infos[1])  

        for i in race_dict:
            if i == infos[2]:
                label['Race']=race_dict[i]     
    labels_utkface.append(label)
with open("root_dir/datasets/labels/utkface_labels.csv","w",newline = '') as csv_file:
    csv_writer = csv.DictWriter(csv_file,fieldnames=headers)
    csv_writer.writeheader()
    csv_writer.writerows(labels_utkface)


with open("root_dir/datasets/labels/utkface_labels.csv","r",newline = '') as csv_file:
    csv_reader = csv.reader(csv_file)
    c=0
    for line in csv_reader:
        c+=1
        if c%500==0:
            print(line)


        


# In[ ]:




