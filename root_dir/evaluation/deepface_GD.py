from deepface import DeepFace
import os
import argparse
from utils import *



def main():
    output_score = MetricsBuilder()
    args = read_args()
    aligned_dataset_path = args.aligned_path
    deidentified__dataset_path  = args.deidentified_path
    files = os.listdir(aligned_dataset_path)
    output_score_file = get_output_filename("deepface", aligned_dataset_path, deidentified__dataset_path)
    f = open(output_score_file, 'w')
    
    device = 'cuda' if True else 'cpu'


    samples:int = len(files)
    succeses:int = 0     
    for file in files: 
        aligned_img_path = os.path.join(aligned_dataset_path, file)
        deidentified_img_path = os.path.join(deidentified__dataset_path, file)
        if not os.path.exists(aligned_img_path):
            print(f"{aligned_dataset_path} does not exist")
            continue
        if not os.path.exists(deidentified_img_path):
            print(f"{deidentified__dataset_path} does not exist")
            continue
        #run the predicctions
        pred_aligned = DeepFace.analyze(img_path = aligned_img_path, actions = ['gender'],enforce_detection=False)
        pred_deidentified = DeepFace.analyze(img_path = deidentified_img_path, actions = ['gender'],enforce_detection=False)
        #Log the result
        #f.writelines(f"{emotion_aligned}, {emotion_deidentified},{True if emotion_aligned == emotion_deidentified else False}")
        output_score.add_output_message(pred_aligned)
        output_score.add_output_message(pred_deidentified)
        gender_aligned = pred_aligned[0].get("dominant_gender", [])#.get("dominant_gender", "-")
        gender_deidentified = pred_deidentified[0].get("dominant_gender", [])#.get("dominant_gender", "--")
        

        #Increase the succeses if are equal
        if gender_aligned == gender_deidentified: 
            succeses+=1
    f.close()
    accuracy= 0
    accuracy = (succeses / samples)*100
    return output_score.add_metric("deepface", "accuracy", "{:1.2f}%".format(accuracy))

if __name__ == "__main__":
    #main()
    result, output, _  =with_no_prints(main)
    print(result.build())
