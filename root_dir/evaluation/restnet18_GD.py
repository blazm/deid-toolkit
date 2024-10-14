import os
import argparse
from utils import *
from data_utility.Restnet18.model import Model

#TODO: change the paths
CHECKPOINT_NAME = "./root_dir/evaluation/data_utility/Restnet18/face_gender_classification_transfer_learning_with_ResNet18.pth"
#python ./root_dir/evaluation/hsemotion.py ./root_dir/datasets/aligned/fri/ ./root_dir/datasets/pixelize/fri/


def main():
    output_score = MetricsBuilder()
    args = read_args()
    #get the mandatory args
    #get the only two params
    aligned_dataset_path = args.aligned_path
    deidentified__dataset_path = args.deidentified_path
    files = os.listdir(aligned_dataset_path)
    output_score_file = get_output_filename("restnet18_GD", aligned_dataset_path, deidentified__dataset_path)
    f = open(output_score_file, 'w')
    
    device = 'cuda' if True else 'cpu'
    model = Model(CHECKPOINT_NAME)


    samples:int = len(files)
    succeses:int = 0 
    
    for file in files: 
        aligned_img_path = os.path.join(aligned_dataset_path, file)
        deidentified_img_path = os.path.join(deidentified__dataset_path, file)
        if not  os.path.exists(aligned_img_path):
            print(f"{aligned_img_path} does not exist")
            continue
        if not  os.path.exists(deidentified_img_path):
            print(f"{deidentified_img_path} does not exist")
            continue
        #convert images
        index_aligned, label_aligned = model.fit(aligned_img_path)
        index_deidentified, label_deidentified = model.fit(deidentified_img_path)
        #run the predicctions
        print(f"{aligned_img_path}: {label_aligned}, {deidentified_img_path}:{label_deidentified}")
        #Log the result
        #f.writelines(f"{emotion_aligned}, {emotion_deidentified},{True if emotion_aligned == emotion_deidentified else False}")
        #Increase the succeses if are equal
        if label_aligned == label_deidentified: 
            succeses+=1
    f.close()
    accuracy= 0
    accuracy = (succeses / samples)*100
    return output_score.add_metric("restnet18", "accuracy", "{:1.2f}%".format(accuracy))

if __name__ == "__main__":
    #main()
    result, _ , _  =with_no_prints(main)
    print(result.build())
