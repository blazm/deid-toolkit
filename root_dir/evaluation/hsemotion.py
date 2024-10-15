import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './data_utility/hsemotion/hsemotion')))
from data_utility.hsemotion.hsemotion.facial_emotions import HSEmotionRecognizer 
import utils as util

import cv2


MODEL_NAME = 'enet_b0_8_best_afew'
#python ./root_dir/evaluation/hsemotion.py ./root_dir/datasets/aligned/fri/ ./root_dir/datasets/pixelize/fri/


def main():
    args = util.read_args()
    aligned_dataset_path = args.aligned_path
    deidentified__dataset_path  = args.deidentified_path
    files = os.listdir(aligned_dataset_path)
    #output_score_file = util.get_output_filename("hsemotion", aligned_dataset_path, deidentified__dataset_path)
    #f = open(output_score_file, 'w')

    path_to_save = args.save_path
    dataset_name = util.get_dataset_name_from_path(aligned_dataset_path)
    technique_name = util.get_technique_name_from_path(deidentified__dataset_path)
    metrics_df= util.Metrics(name_evaluation="hsemotion", 
                              name_dataset=dataset_name,
                              name_technique=technique_name,
                              name_score="isMatch")
    
    
    device = 'cuda' if True else 'cpu'
    fer=HSEmotionRecognizer(model_name=MODEL_NAME,device=device) # device is cpu or gpu
    samples:int = len(files)
    succeses:int = 0 
    for file in files: 
        aligned_img_path = os.path.join(aligned_dataset_path, file)
        deidentified_img_path = os.path.join(deidentified__dataset_path, file)
        if not os.path.exists(aligned_img_path):
            print(f"{aligned_img_path} does not exist")
            continue
        if not os.path.exists(deidentified_img_path):
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
        #f.writelines(f"{emotion_aligned}, {emotion_deidentified},{is_match}")
        #Increase the succeses if are equal
        if emotion_aligned == emotion_deidentified: 
            succeses+=1
        metrics_df.add_score(path_aligned=aligned_img_path,
                             path_deidentified=deidentified_img_path,
                             metric_result=is_match)
        
    #f.close()
    metrics_df.save_to_csv(path_to_save)
    print(f"hsemotion saved into {path_to_save}")

    #accuracy = (succeses / samples)*100
    return

if __name__ == "__main__":
    main()
    #main()