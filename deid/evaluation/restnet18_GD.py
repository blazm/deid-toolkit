import os
import argparse
import sys
from pathlib import Path
import utils as util
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_DATA_UTILITY = _SCRIPT_DIR / "data_utility" / "Restnet18"

try:
    if _DATA_UTILITY.is_dir():
        sys.path.insert(0, str(_SCRIPT_DIR / "data_utility"))
        from data_utility.Restnet18.model import Model  # type: ignore
    else:
        Model = None  # type: ignore
except ModuleNotFoundError:
    Model = None  # type: ignore

SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINT_NAME = str(SCRIPT_DIR / "weights" / "face_gender_classification_transfer_learning_with_ResNet18.pth")
# python ./hsemotion.py ./aligned/fri/ ./pixelize/fri/
labels_map = {"male":1, "female":-1}

def main():
    if Model is None:
        print(
            "RestNet18-GD evaluation skipped: 'data_utility/Restnet18' not found.\n"
            "  Download the model into deid/evaluation/data_utility/Restnet18/"
        )
        return

    args = util.read_args()
    #get the mandatory args
    #get the only two params
    aligned_dataset_path = args.aligned_path
    deidentified_dataset_path = args.deidentified_path
    path_to_save = args.save_path
    path_to_log = args.dir_to_log

    dataset_name = util.get_dataset_name_from_path(aligned_dataset_path)
    technique_name = util.get_technique_name_from_path(deidentified_dataset_path)
    metrics_df= util.Metrics(name_score="isMatch")
    
    files = os.listdir(aligned_dataset_path)
    #output_score_file = util.get_output_filename("restnet18_GD", aligned_dataset_path, deidentified_dataset_path)
    #f = open(output_score_file, 'w')
    
    device = 'cuda' if True else 'cpu'
    model = Model(CHECKPOINT_NAME)
    
    for i, file in enumerate(tqdm(files, total=len(files), desc=f"restnet18 | {dataset_name}-{technique_name} ")):
        if util._TEST_SINGLE and i > 0:
            break
        aligned_img_path = os.path.join(aligned_dataset_path, file)
        deidentified_img_path = os.path.join(deidentified_dataset_path, file)
        if not os.path.exists(aligned_img_path):
            util.log(os.path.join(path_to_log,"resnet18.txt"), 
                     f"({dataset_name}) The source images are not in {aligned_img_path} ")
            print(f"{aligned_dataset_path} does not exist")
            continue
        if not  os.path.exists(deidentified_img_path):
            util.log(os.path.join(path_to_log,"resnet18.txt"), 
                     f"({technique_name}) The deidentified images are not in {deidentified_img_path} ")
            print(f"{deidentified_img_path} does not exist")
            continue
        #convert images
        index_aligned, label_aligned = model.fit(aligned_img_path)
        index_deidentified, label_deidentified = model.fit(deidentified_img_path)
        #run the predicctions
        #Log the result
        #f.writelines(f"{emotion_aligned}, {emotion_deidentified},{True if emotion_aligned == emotion_deidentified else False}")
        #Increase the succeses if are equal
        is_math = 1 if label_aligned == label_deidentified else 0
        metrics_df.add_score(file,is_math)
        metrics_df.add_column_value("aligned_predictions", labels_map[label_aligned])
        metrics_df.add_column_value("deidentified_predictions", labels_map[label_deidentified])
    #f.close()
    metrics_df.save_to_csv(path_to_save)
    print(f"resnet18 scores saved in {path_to_save}")
    #accuracy= 0
    #accuracy = (succeses / samples)*100
    return

if __name__ == "__main__":
    #main()
    main()
