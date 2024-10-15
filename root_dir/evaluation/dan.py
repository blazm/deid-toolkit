import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from data_utility.DAN.networks.dan import DAN
#from utils import * 
import utils as util

AFFECT_NET_PATH = './root_dir/evaluation/data_utility/DAN/checkpoints/affecnet8_epoch5_acc0.6209.pth'
class Model():
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_transforms = transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                                ])
        #TODO: ask blaz about what should I do with the missmatch of the classes
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
def accuracy(attempts, successes): 
    return successes/attempts
def main():
    args = util.read_args()
    aligned_dataset_path = args.aligned_path
    deidentified__dataset_path  = args.deidentified_path

    path_to_save = args.save_path
    dataset_name = util.get_dataset_name_from_path(aligned_dataset_path)
    technique_name = util.get_technique_name_from_path(deidentified__dataset_path)
    metrics_df= util.Metrics(name_evaluation="dan", 
                              name_dataset=dataset_name,
                              name_technique=technique_name,
                              name_score="isMatch")
    
    #output_scores_file = util.get_output_filename("dan", aligned_dataset_path, deidentified__dataset_path)
    #f = open(output_scores_file, 'w')
    files = os.listdir(aligned_dataset_path)

    model = Model() #initialize the model
    samples:int = len(files)
    succeses:int = 0 
    for file in files: 
        aligned_img_path = os.path.join(aligned_dataset_path, file)
        deidentified_img_path = os.path.join(deidentified__dataset_path, file)
        if not os.path.exists(aligned_img_path):
            print(f"{aligned_dataset_path} does not exist")
            continue
        if not  os.path.exists(deidentified__dataset_path):
            print(f"{deidentified__dataset_path} does not exist")
            continue
        #evaluation
        index_aligned, label_aligned = model.fit(aligned_img_path)
        index_deidentified, label_deidentified  = model.fit(deidentified_img_path)
        #log the result
        is_match = 1 if index_aligned == index_deidentified else 0
        #f.writelines(f"{label_aligned}, {label_deidentified},{is_match}")
        #increase the accuracy
        if index_aligned == index_deidentified: 
            succeses+=1
        metrics_df.add_score(path_aligned=aligned_img_path, 
                             path_deidentified=deidentified_img_path,
                             metric_result=(is_match))

    metrics_df.save_to_csv(path_to_save)
    print(f"dan saved into {path_to_save}")

    #f.close()
    #accuracy = (succeses / samples)*100

if __name__ == "__main__":
    main()