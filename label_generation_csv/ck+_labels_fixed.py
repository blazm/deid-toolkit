import os
import csv
from tqdm import tqdm



headers = ['Name', 'Path', 'Identity', 'Gender_code', 'Gender', 'Age', 'Race_code', 'Race', 'date of birth', 'Emotion_code',
           'Neutral', 'Anger', 'Scream', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise', 'Sun glasses',
           'Scarf', 'Eyeglasses', 'Beard', 'Hat', 'Angle']


directory = os.path.abspath(os.path.join("deid-toolkit","root_dir","datasets","aligned","ck+"))
emotion_directory =  os.path.abspath(os.path.join("Emotion"))
labels_ck = []

# Dictionary to map emotion labels to codes
emotion_dict = {'Neutral': 0, 'Anger': 1, 'Scream': 2, 'Contempt': 3, 'Disgust': 4,
                'Fear': 5, 'Happy': 6, 'Sadness': 7, 'Surprise': 8}


img_list = sorted([img_name for img_name in os.listdir(directory) if img_name.lower().endswith(('.png', '.jpg', '.jpeg'))])

image_with_labels =0
total =0 
identities = []
identities_with_labels = []
for img_name in tqdm(img_list, desc="Processing images"):
    total += 1
    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(directory, img_name)
        row = {
            'Name': img_name, 'Path': img_path, 'Identity': '', 'Gender_code': '', 'Gender': '', 'Age': '',
            'Race_code': '', 'Race': '', 'date of birth': '', 'Emotion_code': '', 'Neutral': '', 'Anger': '',
            'Scream': '', 'Contempt': '', 'Disgust': '', 'Fear': '', 'Happy': '', 'Sadness': '', 'Surprise': '',
            'Sun glasses': '', 'Scarf': '', 'Eyeglasses': '', 'Beard': '', 'Hat': '', 'Angle': 0
        }
        imgInfo = img_name.split('_')
        id = imgInfo[0]
        if id not in identities:
            identities.append(id)
        scene = imgInfo[1]
        row['Identity'] = id
        file_dir = os.path.join(emotion_directory, id, scene)
        
        # Comprobación de etiquetas en el directorio
        if os.path.exists(file_dir) and len(os.listdir(file_dir)) > 0:
            file_name_with_labels = os.listdir(file_dir)[0] #get the first file
            file_path_with_labels = os.path.join(file_dir, file_name_with_labels)
            with open(file_path_with_labels, "r") as text_file:
                lines = [line.strip() for line in text_file if line.strip()]
                if lines:
                    emotion_value_int = int(lines[0].split('.')[0])
                    if emotion_value_int == 0:
                        row['Neutral'] = 1
                        row['Emotion_code'] = emotion_dict['Neutral']
                    elif emotion_value_int == 1:
                        row['Anger'] = 1
                        row['Emotion_code'] = emotion_dict['Anger']
                    elif emotion_value_int == 2:
                        row['Contempt'] = 1
                        row['Emotion_code'] = emotion_dict['Contempt']
                    elif emotion_value_int == 3:
                        row['Disgust'] = 1
                        row['Emotion_code'] = emotion_dict['Disgust']
                    elif emotion_value_int == 4:
                        row['Fear'] = 1
                        row['Emotion_code'] = emotion_dict['Fear']
                    elif emotion_value_int == 5:
                        row['Happy'] = 1
                        row['Emotion_code'] = emotion_dict['Happy']
                    elif emotion_value_int == 6:
                        row['Sadness'] = 1
                        row['Emotion_code'] = emotion_dict['Sadness']
                    elif emotion_value_int == 7:
                        row['Surprise'] = 1
                        row['Emotion_code'] = emotion_dict['Surprise']
                    labels_ck.append(row)
                    image_with_labels += 1
                    if id not in identities_with_labels:  # Solo agregar si aún no está
                        identities_with_labels.append(id)

                
        else:
            print(img_path, " deleted!")
            #os.remove(img_path) #remove the file that I don't need 
#output_path = os.path.join("root_dir","datasets","labels","ck+_fix_labels.csv")
#os.makedirs(os.path.dirname(output_path), exist_ok=True)
#
#with open(output_path, "w", newline='') as csv_file:
#    csv_writer = csv.DictWriter(csv_file, fieldnames=headers)
#    csv_writer.writeheader()
#    csv_writer.writerows(labels_ck)
#print(f"Document saved in: {output_path}")
## Read and print some lines from the CSV file to verify
#with open(output_path, "r", newline='') as csv_file:
#    csv_reader = csv.reader(csv_file)
#    c = 0
#    for line in csv_reader:
#        c += 1
#        if c % 500 == 0:
#            print(line)
#
print(f"Total images: {total} | Images with labels {image_with_labels}")
print(f"Porcentaje: {image_with_labels / total}")        
print(f"Total Identities: {len(identities)} - with label: {len(identities_with_labels)}")
