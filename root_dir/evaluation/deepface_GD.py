import tqdm
from deepface import DeepFace
import os
import argparse
import utils as util
from tqdm import tqdm

#import warnings
#warnings.filterwarnings('ignore', category=FutureWarning)


labels_map = {"Man": 0, "Woman":-1}

def main():
    args = util.read_args()
    aligned_dataset_path = args.aligned_path
    deidentified__dataset_path  = args.deidentified_path
    path_to_save = args.save_path
    path_to_log = args.dir_to_log

    files = os.listdir(aligned_dataset_path)
    #output_score_file = util.get_output_filename("deepface", aligned_dataset_path, deidentified__dataset_path)
    #f = open(output_score_file, 'w')

    dataset_name = util.get_dataset_name_from_path(aligned_dataset_path)
    technique_name = util.get_technique_name_from_path(deidentified__dataset_path)
    metrics_df= util.Metrics(name_score="isMatch")
    
    device = 'cuda' if True else 'cpu'


    for file in tqdm(files, total =len(files), desc=f"deepface | {dataset_name}-{technique_name} "): 
        aligned_img_path = os.path.join(aligned_dataset_path, file)
        deidentified_img_path = os.path.join(deidentified__dataset_path, file)
        if not os.path.exists(aligned_img_path):
            util.log(os.path.join(path_to_log,"deepface.txt"), 
                     f"({dataset_name}) The source images are not in {aligned_img_path} ")
            print(f"{aligned_dataset_path} does not exist")
            continue
        if not  os.path.exists(deidentified_img_path):
            util.log(os.path.join(path_to_log,"deepface.txt"), 
                     f"({technique_name}) The deidentified images are not in {deidentified_img_path} ")
            print(f"{deidentified_img_path} does not exist")
            continue
        #run the predicctions
        pred_aligned = DeepFace.analyze(img_path = aligned_img_path, actions = ['gender'],enforce_detection=False)
        pred_deidentified = DeepFace.analyze(img_path = deidentified_img_path, actions = ['gender'],enforce_detection=False)
        #Log the result
        #f.writelines(f"{emotion_aligned}, {emotion_deidentified},{True if emotion_aligned == emotion_deidentified else False}")
        gender_aligned = pred_aligned[0].get("dominant_gender", [])#.get("dominant_gender", "-")
        gender_deidentified = pred_deidentified[0].get("dominant_gender", [])#.get("dominant_gender", "--")
        
        #Increase the succeses if are equal
        
        is_match = 1 if gender_aligned == gender_deidentified else 0
        metrics_df.add_score(img=file,metric_result=is_match)
        metrics_df.add_column_value("aligned_predictions", labels_map[gender_aligned])
        metrics_df.add_column_value("deidentified_predictions", labels_map[gender_deidentified])
    #f.close()
    metrics_df.save_to_csv(path_to_save)
    print(f"deepface saved into {path_to_save}")

    return

if __name__ == "__main__":
    #main()
    main()