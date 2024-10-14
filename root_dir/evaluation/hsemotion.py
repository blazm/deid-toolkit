from data_utility.hsemotion.hsemotion.facial_emotions import HSEmotionRecognizer 
import os
import argparse
from utils import *
import cv2


MODEL_NAME = 'enet_b0_8_best_afew'
#python ./root_dir/evaluation/hsemotion.py ./root_dir/datasets/aligned/fri/ ./root_dir/datasets/pixelize/fri/


def main():
    output_score = MetricsBuilder()
    args = read_args()
    aligned_dataset_path = args.aligned_path
    deidentified__dataset_path  = args.deidentified_path
    files = os.listdir(aligned_dataset_path)
    output_score_file = get_output_filename("hsemotion", aligned_dataset_path, deidentified__dataset_path)
    f = open(output_score_file, 'w')
    
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
        f.writelines(f"{emotion_aligned}, {emotion_deidentified},{True if emotion_aligned == emotion_deidentified else False}")
        #Increase the succeses if are equal
        if emotion_aligned == emotion_deidentified: 
            succeses+=1
    f.close()
    accuracy = (succeses / samples)*100
    return output_score.add_metric("hsemotion", "accuracy", "{:1.2f}%".format(accuracy))

if __name__ == "__main__":
    result, _ , _  =with_no_prints(main)
    print(result.build())
    #main()