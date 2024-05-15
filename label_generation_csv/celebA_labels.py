#!/usr/bin/env python
# coding: utf-8

# ## .CSV

# In[5]:


# import csv

# input_txt_path = r"root_dir/datasets/original/celeba/CelebAMask-HQ-attribute-anno.txt"
# output_csv_path = r"root_dir/datasets/original/celeba/CelebAMask-HQ-attribute-anno.csv"

# with open(input_txt_path, "r") as txt_file:
#     lines = txt_file.readlines()

# # Ignore the first line
# lines = lines[1:]

# with open(output_csv_path, "w", newline='') as csv_file:
#     csv_writer = csv.writer(csv_file)

#     for line in lines:
#         # Split the line into fields based on spaces or other delimiters as needed
#         fields = line.strip().split()
#         csv_writer.writerow(fields)

# print(f"Converted {input_txt_path} to {output_csv_path}, ignoring the first line.")


# In[6]:


import os
import csv
import numpy as np

headers = ['Name', 'Path', 'Identity', 'Gender_code', 'Gender', 'Age', 'Race_code', 'Race', 'date of birth', 'Emotion_code',
           'Neutral', 'Anger', 'Scream', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise',
           'Sun glasses', 'Scarf', 'Eyeglasses', 'Beard', 'Hat', 'Angle']

directory = r"root_dir/datasets/original/celeba/img"
labels_celeba = []
emotion_dict = {'Neutral': 0, 'Anger': 1, 'Scream': 2, 'Contempt': 3, 'Disgust': 4,
                'Fear': 5, 'Happy': 6, 'Sadness': 7, 'Surprise': 8}

mapping_path = r"root_dir/datasets/original/celeba/CelebA-HQ-to-CelebA-mapping.txt"
identity_path = r"root_dir/datasets/original/celeba/identity_CelebA.txt"
attribute_csv_file = r"root_dir/datasets/original/celeba/CelebAMask-HQ-attribute-anno.csv"

# Function to load data from a text file into a dictionary
def load_data(file_path):
    data = {}
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 2:  # Ensure each line has 2 columns
                data[parts[0]] = parts[1]
            elif len(parts) == 3:  # Specific case for the first file
                data[parts[2]] = (parts[0], parts[1])  # Use the third column as key
    return data

def find_id_from_idx(idx, mapping_data, identity_data):
    # Find the corresponding orig_file in the first file
    for orig_file, (file_idx, _) in mapping_data.items():
        if file_idx == idx:
            # Check if the orig_file exists in the second file
            if orig_file in identity_data:
                return identity_data[orig_file]
            else:
                return "Unknown"
    return "Unknown"

def find_attribute_value(image_name, attribute_column, csv_file):
    with open(csv_file, "r", newline="") as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)
        # Check if the specified attribute column exists in the headers
        if attribute_column not in headers:
            print(f"The column '{attribute_column}' does not exist in the CSV file.")
            return None
        attribute_index = headers.index(attribute_column)
        for row in reader:
            # Check if the first column (image name) matches the given image
            if row[0] == image_name:
                return row[attribute_index]
    # If the image is not found in the CSV file
    print(f"The image '{image_name}' was not found in the CSV file.")
    return None

# Load data from the first file (orig_file -> (idx, idx_orig))
mapping_data = load_data(mapping_path)

# Load data from the second file (orig_file -> id)
identity_data = load_data(identity_path)

c = 0
nb_img = len(os.listdir(directory))
for img_name in os.listdir(directory):
    c += 1
    progression = np.round(100 * c / nb_img, 3)
    print(f"\n Progression: {progression}% \n")

    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        idx = img_name.split('.')[0]
        img_id = find_id_from_idx(idx, mapping_data=mapping_data, identity_data=identity_data)
        img_path = os.path.join(directory, img_name)

        label = {
            'Name': img_name, 'Path': img_path, 'Identity': img_id, 'Gender_code': '', 'Gender': '', 'Age': '', 'Race_code': '', 'Race': '', 'date of birth': '',
            'Emotion_code': '', 'Neutral': '', 'Anger': '', 'Scream': '', 'Contempt': '', 'Disgust': '',
            'Fear': '', 'Happy': '', 'Sadness': '', 'Surprise': '', 'Sun glasses': '',
            'Scarf': '', 'Eyeglasses': '', 'Beard': '', 'Hat': '', 'Angle': ''
        }
        
        desired_attributes = ['Eyeglasses', 'Male', 'No_Beard', 'Wearing_Hat']

        for attr in desired_attributes:
            attribute_value = find_attribute_value(img_name, attr, attribute_csv_file)
            if attribute_value is not None:
                attribute_value = int(attribute_value)
                if attr == 'Eyeglasses':
                    label['Eyeglasses'] = attribute_value
                elif attr == 'Male':
                    label['Gender'] = 'Male' if attribute_value == 1 else 'Female'
                    label['Gender_code'] = attribute_value
                elif attr == 'No_Beard':
                    label['Beard'] = -attribute_value
                elif attr == 'Wearing_Hat':
                    label['Hat'] = attribute_value
        
        labels_celeba.append(label)

output_path = r"root_dir/datasets/labels/celeba_labels.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w", newline='') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=headers)
    csv_writer.writeheader()
    csv_writer.writerows(labels_celeba)

with open(output_path, "r", newline='') as csv_file:
    csv_reader = csv.reader(csv_file)
    c = 0
    for line in csv_reader:
        c += 1
        if c % 500 == 0:
            print(line)


# In[ ]:




