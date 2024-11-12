import argparse
import os
import shutil
import pandas as pd
import tarfile

# remove the mixed expression if we don't wnat the part two
COLUMNS = [
    "Name",
    "Path",
    "Identity",
    "Gender_code",
    "Gender",
    "Age",
    "Race_code",
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
emotionMap: dict = {
    "anger": "Anger",
    "disgust": "Disgust",
    "fear": "Fear",
    "happiness": "Happy",
    "neutral": "Neutral",
    "sadness": "Sadness",
    "surprise": "Surprise",
}
codeMap:dict =  {
    "anger": "1",
    "disgust": "4",
    "fear": "5",
    "happiness": "6",
    "neutral": "0",
    "sadness": "7",
    "surprise": "8",
}
image_extensions = {".jpg", ".jpeg", ".png"}

# Create a dataframe to save subject_emotion_take_image.png
# create a option to get the first take  (takelist[0])

# creata a func extract only number given a string


def extract(directory):
    # extract all the tar files if not already extracted
    for filename in os.listdir(directory):
        if filename.endswith(".tar"):  # Check if tar file
            file_path = os.path.join(directory, filename)
            if not os.path.isdir(file_path.replace(".tar", "")):
                # If not exist extract otherwise pass
                with tarfile.open(file_path) as tar:
                    extract_path = os.path.join(
                        directory, os.path.splitext(filename)[0]
                    )
                    os.makedirs(extract_path, exist_ok=True)
                    # extract all the content
                    tar.extractall(path=extract_path)
                    print(f"Extracted {filename} to {extract_path}")
            else:
                print(f"{filename} already extracted")


def copyImg(src: str, destination: str, filename: str) -> str:
    """Copy the image from the source to the destination

    Args:
        src (str): source image path
        destination (str): destination directory
        filename (str): name of the new saved file

    Returns:
        str: the relative path of the saved file
    """
    # files[selected_index], args.to , subj_emo_take_image.jpg
    destination_file = os.path.join(destination, filename)
    shutil.copy2(src, destination_file)
    print("saved: ", filename)
    return destination_file


def get_emotion_from_path(path: str) -> str:
    # get the emotion from the path of the image
    return path.split("/")[-2]


def get_subject_from_path(path: str) -> str:
    # -4 because some of subjects has 2 sessions
    # subject/subject/session/take/image.jpg
    return path.split("/")[-4]


def process_images(datasetDir, destinationDir):
    data = []
    for root, dirs, files in os.walk(datasetDir):
        # filter images files only
        # Filter and sort image files by their numeric suffix (onyly images files)
        image_files = sorted(
            [f for f in files if f.lower().endswith((".jpg", ".png"))],
            key=lambda x: int(x.split("_")[-1].split(".")[0]),
        )
        if len(image_files) > 0:  # if this is the last directory where the images are
            print("Files: ", image_files, " len: ", len(image_files))
            emotion: str = get_emotion_from_path(root)
            # emotion remove [:3 ] if we want the full name
            if emotionMap.get(emotion) not in COLUMNS:
                # check if the emotion is in the list of emotions first before extract the attributes for performance
                print(f" skip {emotion}, not in COLUMNS")
                continue

            selected_file: str = image_files[
                len(image_files) // 2
            ]  # Select the middle image
            take = root.split("/")[-1]  # takeNumber
            subject = get_subject_from_path(root)
            imgNameCopy = f"{subject}_{emotion[:3]}_{take}_{selected_file}"
            destinationFilename = copyImg(
                os.path.join(root, selected_file),
                destinationDir,
                imgNameCopy,
            )
            emotion_values = []
            for e in COLUMNS[3:]:
                if e == emotionMap.get(emotion):
                    emotion_values.append("1")
                elif e == "Emotion_code":
                    emotion_values.append(codeMap.get(emotion))
                else:
                    emotion_values.append("")
             

            data.append(
                [
                    imgNameCopy, #name
                    os.path.join(root, destinationFilename), #path
                    subject, # identity
                    *emotion_values,
                ]
            )

    df = pd.DataFrame(data, columns=COLUMNS)
    df.to_csv(os.path.join(destinationDir, "mug_labeled.csv"), index=False)


def process(args):
    directory = args.path
    destination = args.to

    extract(os.path.join(directory, "subjects3"))
    #
    os.makedirs(destination, exist_ok=True)
    process_images(
        datasetDir=os.path.join(directory, "subjects3"),
        destinationDir=destination,
    )
    print("----------------" * 2)
    print("All data is labeled and saved in the output directory")
    print("----------------" * 2)
    return
def test(args):
    directory = args.path
    df = pd.read_csv(directory)
    for index, row in df.iterrows():
        imageName = row.iloc[0]
        for col in df.columns[1:]:  
            if row[col] == 1:
                print(f"Imagen: {imageName} | Column: {col} | Emotion code: {row['Emotion_code']}")
def changeDir(args):
    currDir = args.path
    newDir = args.changecsvdir
    df = pd.read_csv(currDir)
    # Change route file
    for index, row in df.iterrows():
        path = row['Path']
        fileName = path.split('/')[-1]  #Get file name and store it
        newPath = os.path.join(newDir, fileName)  # Create new path
        df.at[index, 'Path'] = newPath  # update the path
    
    # Guardar el DataFrame modificado en un nuevo archivo CSV
    df.to_csv(currDir, index=False)
    print(f"CSV saved in: {currDir}")



def isValidArgs(args) -> bool:
    if not (os.path.isdir(args.path) or os.path.isfile(args.path)):
        print("The directory or file provided does not exist")
        return False
    if  args.test and not os.path.isfile(args.path) and not args.path.endswith(".csv"):
        print("The path provided is a file, not a csv")
        return False
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to process MUG dataset, extract most expressive faces"
    )
    parser.add_argument(
        "path", type=str, help="Path where the dataset is located, e.g., /../MUG"
    )
    parser.add_argument(
        "--to",
        type=str,
        default="./mug-still",
        help="Directory where the CSV output will be saved",
    )
 
    parser.add_argument(
        "-test",
        action="store_true", 
        help="provide the mug-still images directory and test the csv file",
    )
    parser.add_argument(
        "--changecsvdir", 
        help="change the labeled directory from the csv file",
    )
    args = parser.parse_args()
    if isValidArgs(args):
        print("Everything is ready!")
        if args.test:
            print("testing: ",args.test)
            test(args)
        elif args.changecsvdir:
            print("changing directory from csv: ",args.changecsvdir)
            changeDir(args)
        else:
            print("processing...")
            process(args)
