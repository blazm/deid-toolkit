from deepface import DeepFace
import os
import argparse
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = argparse.ArgumentParser(description="Evaluate deepface facial expression between aligned and deidentified images")
    parser.add_argument('path', type=str, nargs=2,
                        help=('Paths of the aligned and deidentified datasets')) 
    args = parser.parse_args()
    assert os.path.exists(args.path[0])
    assert os.path.exists(args.path[1])
    return args.path[0], args.path[1]

def main():
    output_score = MetricsBuilder()
    aligned_dataset_path, deidentified__dataset_path = parse_args()
    files = os.listdir(aligned_dataset_path)
    output_score_file = get_output_filename("deepface", aligned_dataset_path, deidentified__dataset_path)
    f = open(output_score_file, 'w')
    
    device = 'cuda' if True else 'cpu'


    samples:int = len(files)
    succeses:int = 0 
    
    for file in files: 
        aligned_img_path = os.path.join(aligned_dataset_path, file)
        deidentified_img_path = os.path.join(deidentified__dataset_path, file)
        assert os.path.exists(aligned_img_path)
        assert os.path.exists(deidentified_img_path)
        #run the predicctions
        pred_aligned = DeepFace.analyze(img_path = aligned_img_path, actions = ['gender'],enforce_detection=False)
        pred_deidentified = DeepFace.analyze(img_path = deidentified_img_path, actions = ['gender'],enforce_detection=False)
        #Log the result
        #f.writelines(f"{emotion_aligned}, {emotion_deidentified},{True if emotion_aligned == emotion_deidentified else False}")
        print(f"{aligned_img_path}: {pred_aligned}, {deidentified_img_path}:{pred_deidentified}")
        #Increase the succeses if are equal
        if pred_aligned == pred_deidentified: 
            succeses+=1
    f.close()
    accuracy= 0
    #accuracy = (succeses / samples)*100
    return output_score.add_metric("deepface", "accuracy", "{:1.2f}%".format(accuracy))

if __name__ == "__main__":
    main()
    #result, _ , _  =with_no_prints(main)
    #print(result.build())
