import os
import argparse
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm

# Graceful fallback: data_utility is an external dependency that may be missing
try:
    from data_utility.DAN.networks.dan import DAN  # type: ignore
except ModuleNotFoundError:
    DAN = None  # type: ignore

import utils as util

#Emotion_code: 0 = Neutral, 1 = Anger, 2 = Scream, 3 = Contempt, 4 = Disgust, 5 = Fear, 6 = Happy, 7 = Sadness, 8 = Surprise
#this is important to keep consistence with the toolkit
labels_map= {"neutral":0, "happy":6, "sad":7,"surprise":8, "fear":5, "disgust":4,"anger":1,"contempt":3}

SCRIPT_DIR = Path(__file__).resolve().parent
AFFECT_NET_PATH = str(SCRIPT_DIR / "weights" / "affecnet8_epoch5_acc0.6209.pth")
class Model():
    def __init__(self):
        _force_cpu = os.environ.get("DEID_FORCE_CPU", "0") in ("1", "true", "yes")
        _has_cuda = not _force_cpu and torch.cuda.is_available() and torch.cuda.device_count() > 0
        self.device = torch.device("cuda:0" if _has_cuda else "cpu")
        self.data_transforms = transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                                ])
        self.labels = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt']
        self.model = DAN(num_head=4, num_class=8)
        #checkpoint = torch.load('./checkpoints/affecnet8_epoch6_acc0.6326.pth',
        #    map_location=self.device)
        checkpoint = torch.load(AFFECT_NET_PATH,
            map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        self.model.to(self.device)
        self.model.eval()
    
    def fit(self, path):
        img = Image.open(path).convert('RGB')
        img = self.data_transforms(img)
        img = img.view(1,3,224,224)
        img = img.to(self.device)
        with torch.set_grad_enabled(False):
            out, _, _ = self.model(img)
            _, pred = torch.max(out,1)
            index = int(pred)
            label = self.labels[index]

            return index, label

def main():
    if DAN is None:
        print(
            "DAN evaluation skipped: 'data_utility' not found.\n"
            "  Download AffecNet checkpoint into deid/evaluation/weights/affecnet8_epoch5_acc0.6209.pth"
        )
        return

    args = util.read_args()
    aligned_dataset_path = args.aligned_path
    deidentified__dataset_path  = args.deidentified_path
    path_to_log = args.dir_to_log

    path_to_save = args.save_path
    dataset_name = util.get_dataset_name_from_path(aligned_dataset_path)
    technique_name = util.get_technique_name_from_path(deidentified__dataset_path)
    metrics_df= util.Metrics(name_score="isMatch")

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    files = [f for f in os.listdir(aligned_dataset_path) if Path(f).suffix.lower() in valid_extensions]

    model = Model() #initialize the model
    for i, file in enumerate(tqdm(files,total= len(files),  desc=f"dan | {dataset_name}-{technique_name}")):
        if util._TEST_SINGLE and i > 0:
            break
        aligned_img_path = os.path.join(aligned_dataset_path, file)
        deidentified_img_path = os.path.join(deidentified__dataset_path, file)
        if not os.path.exists(aligned_img_path):
            util.log(os.path.join(path_to_log,"dan.txt"), 
                     f"({dataset_name}) The source images are not in {aligned_img_path} ")
            print(f"{aligned_dataset_path} does not exist")
            continue
        if not  os.path.exists(deidentified_img_path):
            util.log(os.path.join(path_to_log,"dan.txt"), 
                     f"({technique_name}) The deidentified images are not in {deidentified_img_path} ")
            print(f"{deidentified_img_path} does not exist")
            continue
        #evaluation
        index_aligned, label_aligned = model.fit(aligned_img_path)
        index_deidentified, label_deidentified  = model.fit(deidentified_img_path)
        #increase the accuracy
        is_match = 1 if index_aligned == index_deidentified else 0
        
        metrics_df.add_score(img=file, 
                             metric_result=(is_match))
        metrics_df.add_column_value("aligned_predictions", labels_map[label_aligned.lower()])
        metrics_df.add_column_value("deidentified_predictions", labels_map[label_deidentified.lower()])

    metrics_df.save_to_csv(path_to_save)
    print(f"dan saved into {path_to_save}")

if __name__ == "__main__":
    main()