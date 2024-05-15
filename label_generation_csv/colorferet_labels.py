#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import csv
import numpy as np
headers=['Name','Path','Identity','Gender_code','Gender','Age','Race_code','Race','date of birth','Emotion_code',
         'Neutral','Anger','Scream','Contempt','Disgust','Fear','Happy','Sadness','Surprise',
         'Sun glasses','Scarf','Eyeglasses','Beard','Hat','Angle']

directory = "root_dir/datasets/original/colorferet/img"
ground_truths_directory="root_dir/datasets/original/colorferet/ground_truths/name_value"

labels_colorferet =[]

'''frontal -> angle = 0 ; head turned to the left -> angle > 0 ; 
head turned to the right -> angle < 0'''
pose_angle_dict ={'fa':'0','fb':'0','pl':'90','hl':'67,5','ql':'22,5','pr':'-90',
               'hr':'-67,5','qr':'-22,5',
               'ra':'45','rb':'15','rc':'-15','rd':'-45','re':'-75'}

c=0
for img_name in os.listdir(directory):

    c+=1
    nb_img = len(os.listdir(directory))
    progression = np.round(100*c/nb_img,2)
    print(f"\n Progession: {progression}% \n") 

    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(directory, img_name)
        
        infos = img_name.split('_')
        id = infos[0]
        label = {
        'Name':'','Path':'','Identity': '','Gender_code':'','Gender': '', 'Age': '','Race_code':'',
        'Race': '', 'date of birth': '','Emotion_code':'','Neutral': '', 'Anger': '', 'Scream': '',
        'Contempt': '', 'Disgust': '','Fear': '', 'Happy': '', 'Sadness': '', 'Surprise': '', 
        'Sun glasses': '','Scarf': '', 'Eyeglasses': '', 'Beard': '', 'Hat': '', 'Angle': ''
    } 
        if len(infos)==4:
            optionnal_flag=infos[3].split('.')[0]
            pose_name = infos[2]
            if optionnal_flag == 'a' or optionnal_flag == c:
                label['Eyeglasses']=1
            else:
                label['Eyeglasses']=-1
        else:
            pose_name = infos[2].split('.')[0]
        
        date_capture =infos[1] #YYMMDD
        capture_year= int('19'+date_capture[0:2])
        text_file_path = os.path.join(ground_truths_directory, id+'.txt')
        with open (text_file_path,'r') as file: #find the yob in the .txt label
            lines=file.readlines()
        year_of_birth = int(lines[2].split('=')[1])
        age = capture_year-year_of_birth 

        race = lines[3].split('=')[1][:-1] #find the race in the .txt label
        #can be 'Asian,Pacific Islander, Hispanic,Other,Native American,
        # Asian Southern,Asian-Middle-Eastern,Black-or-African-American'
        if race == 'Black-or-African-American':
            race = 'Black'

        gender = lines[1].split('=')[1]
         
        if gender[0:4] == 'Male':
            gender_code=1
            gender = gender[0:4]#there is an unwanted character at the end idk why so we delete it 
        else:
            gender_code=-1
            gender = gender[0:6]
        
        for i in pose_angle_dict:
            if i == pose_name:
                label['Angle']=pose_angle_dict[i]
            
        label = {
        'Name':img_name,'Path':img_path,'Identity': id,'Gender_code':gender_code,'Gender': gender, 
        'Age': age, 'Race_code':'','Race': race, 'date of birth': year_of_birth,
        'Emotion_code':'','Neutral': '', 'Anger': '','Scream': '', 'Contempt': '', 'Disgust': '',
        'Fear': '', 'Happy': '', 'Sadness': '', 'Surprise': '', 'Sun glasses': '',
        'Scarf': '', 'Eyeglasses': '', 'Beard': '', 'Hat': '', 'Angle': ''
    } 
        
    labels_colorferet.append(label)



with open("root_dir/datasets/labels/colorferet_labels.csv","w",newline = '') as csv_file:
    csv_writer = csv.DictWriter(csv_file,fieldnames=headers)
    csv_writer.writeheader()
    csv_writer.writerows(labels_colorferet)


with open("root_dir/datasets/labels/colorferet_labels.csv","r",newline = '') as csv_file:
    csv_reader = csv.reader(csv_file)
    c=0
    for line in csv_reader:
        c+=1
        if c%500==0:
            print(line)
        


# In[ ]:


# with open ('root_dir\datasets\original\colorferet\ground_truths\\name_value\\00001.txt','r') as file:
#     lines=file.readlines()
# gender = lines[1].split('=')[1]
# if gender[0:4] == 'Male':
#     gender_code=1
# else:
#     gender_code=-1

# print(gender,gender_code)

