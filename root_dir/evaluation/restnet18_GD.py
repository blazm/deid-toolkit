import os
import argparse
import utils as util
from data_utility.Restnet18.model import Model

#TODO: change the paths
CHECKPOINT_NAME = "./root_dir/evaluation/data_utility/Restnet18/face_gender_classification_transfer_learning_with_ResNet18.pth"
#python ./root_dir/evaluation/hsemotion.py ./root_dir/datasets/aligned/fri/ ./root_dir/datasets/pixelize/fri/


def main():
    args = util.read_args()
    #get the mandatory args
    #get the only two params
    aligned_dataset_path = args.aligned_path
    deidentified_dataset_path = args.deidentified_path
    path_to_save = args.save_path
    dataset_name = util.get_dataset_name_from_path(aligned_dataset_path)
    technique_name = util.get_technique_name_from_path(deidentified_dataset_path)
    metrics_df= util.Metrics(name_evaluation="resnet18", 
                              name_dataset=dataset_name,
                              name_technique=technique_name,
                              name_score="isMatch")
    
    files = os.listdir(aligned_dataset_path)
    #output_score_file = util.get_output_filename("restnet18_GD", aligned_dataset_path, deidentified_dataset_path)
    #f = open(output_score_file, 'w')
    
    device = 'cuda' if True else 'cpu'
    model = Model(CHECKPOINT_NAME)


    samples:int = len(files)
    succeses:int = 0 
    
    for file in files: 
        aligned_img_path = os.path.join(aligned_dataset_path, file)
        deidentified_img_path = os.path.join(deidentified_dataset_path, file)
        if not os.path.exists(aligned_img_path):
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
        is_math = 1 if label_aligned == label_deidentified else 0
        if label_aligned == label_deidentified: 
            succeses+=1
        metrics_df.add_score(aligned_img_path, deidentified_img_path,is_math)
    #f.close()
    metrics_df.save_to_csv(path_to_save)
    print(f"resnet18 scores saved in {path_to_save}")
    #accuracy= 0
    #accuracy = (succeses / samples)*100
    return

if __name__ == "__main__":
    #main()
    main()
