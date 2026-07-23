import sys
import os
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_DATA_UTILITY = _SCRIPT_DIR / "data_utility" / "hsemotion" / "hsemotion"

if _DATA_UTILITY.is_dir():
    sys.path.insert(0, str(_SCRIPT_DIR / "data_utility"))
    from data_utility.hsemotion.hsemotion.facial_emotions import HSEmotionRecognizer  # type: ignore
else:
    HSEmotionRecognizer = None  # type: ignore

import utils as util
from tqdm import tqdm
import cv2


MODEL_NAME = 'enet_b0_8_best_afew'
# python hsemotion.py ./aligned/fri/ ./pixelize/fri/

#Emotion_code: 0 = Neutral, 1 = Anger, 2 = Scream, 3 = Contempt, 4 = Disgust, 5 = Fear, 6 = Happy, 7 = Sadness, 8 = Surprise
#Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, or Surprise
#this is important to keep consistence with the toolkit
labels_map= {"Neutral":0, "Happiness":6, "Sadness":7,"Surprise":8, "Fear":5, "Disgust":4,"Anger":1,"Contempt":3}

def main():
    if HSEmotionRecognizer is None:
        print(
            "HSEmotion evaluation skipped: 'data_utility/hsemotion' not found.\n"
            "  Download HSEmotion model into deid/evaluation/data_utility/hsemotion/hsemotion/"
        )
        return

    args = util.read_args()
    aligned_dataset_path = args.aligned_path
    deidentified__dataset_path  = args.deidentified_path
    files = os.listdir(aligned_dataset_path)
    path_to_log = args.dir_to_log

    #output_score_file = util.get_output_filename("hsemotion", aligned_dataset_path, deidentified__dataset_path)
    #f = open(output_score_file, 'w')

    path_to_save = args.save_path
    dataset_name = util.get_dataset_name_from_path(aligned_dataset_path)
    technique_name = util.get_technique_name_from_path(deidentified__dataset_path)
    metrics_df= util.Metrics( name_score="isMatch")
    
    
    device = 'cuda' if True else 'cpu'
    fer=HSEmotionRecognizer(model_name=MODEL_NAME,device=device) # device is cpu or gpu
    for i, file in enumerate(tqdm(files, total=len(files), desc=f"hsemotion | {dataset_name}-{technique_name}")):
        if util._TEST_SINGLE and i > 0:
            break
        aligned_img_path = os.path.join(aligned_dataset_path, file)
        deidentified_img_path = os.path.join(deidentified__dataset_path, file)
        if not os.path.exists(aligned_img_path):
            util.log(os.path.join(path_to_log,"hsemotion.txt"), 
                     f"({dataset_name}) The source images are not in {aligned_img_path} ")
            print(f"{aligned_dataset_path} does not exist")
            continue
        if not  os.path.exists(deidentified_img_path):
            util.log(os.path.join(path_to_log,"hsemotion.txt"), 
                     f"({technique_name}) The deidentified images are not in {deidentified_img_path} ")
            print(f"{deidentified_img_path} does not exist")
            continue
        #convert images
        align_img= cv2.imread(aligned_img_path)
        deid_img= cv2.imread(deidentified_img_path)
        align_img = cv2.cvtColor(align_img, cv2.COLOR_BGR2RGB)
        deid_img = cv2.cvtColor(deid_img, cv2.COLOR_BGR2RGB)
        #run the predicctions
        emotion_aligned,_=fer.predict_emotions(align_img,logits=True) #
        emotion_deidentified,_=fer.predict_emotions(deid_img,logits=True)
        #Log the result
        is_match = 1 if emotion_aligned == emotion_deidentified else 0
        #Increase the succeses if are equal
        metrics_df.add_score(img=file,metric_result=is_match)
        metrics_df.add_column_value("aligned_predictions", labels_map[emotion_aligned])
        metrics_df.add_column_value("deidentified_predictions", labels_map[emotion_deidentified])
        
    metrics_df.save_to_csv(path_to_save)
    print(f"hsemotion saved into {path_to_save}")

    return

if __name__ == "__main__":
    main()
    #main()