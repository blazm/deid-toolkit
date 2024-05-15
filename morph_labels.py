import os
import csv
import numpy as np
headers = [
    "Name",
    "Path",
    "Identity",
    "Gender_code",
    "Gender",
    "Age",
    'Race_code',
    "Race",
    "date of birth",
    "Emotion_code",
    "Neutral",
    "Anger",
    "Scream",
    "Contempt",
    "Disgust",
    "Fear",
    "Happy",
    "Sadness",
    "Surprise",
    "Sun glasses",
    "Scarf",
    "Eyeglasses",
    "Beard",
    "Hat",
    "Angle",
]

directory = "root_dir/datasets/original/morph/img"
csv_file_out = "root_dir/datasets/labels/morph_labels.csv"
all_data = "root_dir/datasets/original/morph/excel/all_data.csv"

# max_iteration = 500
# threshold = -1

labels_morph = []
c = 0
for img_name in os.listdir(directory):
    # if c >= max_iteration:
    #     continue
    # if c<threshold:
    #     c+=1
    #     continue
    c += 1

    nb_img = len(os.listdir(directory))
    progression = np.round(100*c/nb_img,3)
    print(f"\n Progession: {progression}% \n") 

    if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(directory, img_name)

        infos = img_name.split(".")[0]
        
        id = infos.split("_")[0]
        label = {
            "Name": img_name,
            "Path": img_path,
            "Identity": id,
            "Gender_code": "",
            "Gender": "",
            "Age": "",
            'Race_code':'',
            "Race": "",
            "date of birth": "",
            "Emotion_code": "",
            "Neutral": "",
            "Anger": "",
            "Scream": "",
            "Contempt": "",
            "Disgust": "",
            "Fear": "",
            "Happy": "",
            "Sadness": "",
            "Surprise": "",
            "Sun glasses": "",
            "Scarf": "",
            "Eyeglasses": "",
            "Beard": "",
            "Hat": "",
            "Angle": "",
        }

        
        if len(infos.split('_')[1]) == 4:
            gender = infos.split('_')[1][1]
            age = infos.split('_')[1][2:]
        else:
            gender = infos.split('_')[1][2]
            age = infos.split('_')[1][3:]
        
        if len(age)!=2:
            print("Error age:",age)
    
        if gender=='M':
            gender='Male'
            gender_code=1
        elif gender == 'F':
            gender='Female'
            gender_code=-1
        else:
            print('Error gender:',gender)
        
        label['Gender']=gender
        label['Gender_code'] =gender_code
        label['Age'] = age



        if id[:2] == "00":
            id_to_search = id[2:]
        elif id[0] == "0":
            id_to_search = id[1:]
        else:
            id_to_search = id

        with open(all_data, "r", newline="") as file:
            reader = csv.DictReader(file)
            fieldnames = reader.fieldnames
    
            # Vérifier si le BOM est présent dans la première clé
            if fieldnames[0].startswith('\ufeff'):
                fieldnames[0] = fieldnames[0][1:]
            #print(fieldnames)

            for row in reader:
                #print(row)
                if row['id_num'] == id_to_search:
                    race = row["race"]
                    date_of_birth = row["dob"]
                    facial_hair = row["facial_hair"]
                    glasses = row["glasses"]

                    if race == "W" or race == "White":
                        label["Race"] = "White"
                    elif race == "B" or race=="African-American.Black":
                        label["Race"] = "Black"
                    elif race == "I":
                        label["Race"] = "Indian"
                    elif race == "H":
                        label["Race"] = "Hispanic"
                    elif race == "O" or race == "other":
                        label["Race"] = "Other"
                    elif race == "A":
                        label["Race"] = "Asian"
                    else:
                        print("error colored race:", race)
                    
                    if len(date_of_birth.split("-")[0]) == 4:
                        date_of_birth = date_of_birth.split("-")
                        date_of_birth = (
                            date_of_birth[1]
                            + "."
                            + date_of_birth[2]
                            + "."
                            + date_of_birth[0]
                        )
                    elif(date_of_birth[2]=='/'):
                        date_of_birth=date_of_birth.split('/')
                        date_of_birth = (date_of_birth[1]+'.'+date_of_birth[0]+'.'+ date_of_birth[2])
                    label['date of birth']=date_of_birth

                    if glasses == "0":
                        label["Eyeglasses"] = -1
                    elif glasses == "1":
                        label["Eyeglasses"] = 1

                    if facial_hair == "0":
                        label["Beard"] = 1
                    elif facial_hair =="1":
                        label["Beard"] = -1
        labels_morph.append(label)
                
                
with open("root_dir/datasets/labels/morph_labels.csv", "w", newline="") as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=headers)
    csv_writer.writeheader()
    csv_writer.writerows(labels_morph)

with open("root_dir/datasets/labels/morph_labels.csv", "r", newline="") as csv_file:
    csv_reader = csv.reader(csv_file)
    c = 0
    for line in csv_reader:
        c += 1
        if c % 500 == 0:
            print(line)