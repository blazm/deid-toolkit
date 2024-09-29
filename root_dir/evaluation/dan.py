import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from data_utility.DAN.networks.dan import DAN
from utils import * 

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



def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = argparse.ArgumentParser(description="Evaluate DAN facial expression between aligned and deidentified images")
    parser.add_argument('path', type=str, nargs=2,
                        help=('Paths of the aligned and deidentified datasets')) 
    args = parser.parse_args()
    assert os.path.exists(args.path[0])
    assert os.path.exists(args.path[1])
    return args.path[0], args.path[1]
def accuracy(attempts, successes): 
    return successes/attempts
def main():
    output_score = MetricsBuilder()
    aligned_dataset_path, deidentified__dataset_path = parse_args()
    output_scores_file = get_output_filename("dan", aligned_dataset_path, deidentified__dataset_path)
    f = open(output_scores_file, 'w')
    files = os.listdir(aligned_dataset_path)

    model = Model() #initialize the model
    samples:int = len(files)
    succeses:int = 0 
    for file in files: 
        aligned_img_path = os.path.join(aligned_dataset_path, file)
        deidentified_img_path = os.path.join(deidentified__dataset_path, file)
        assert os.path.exists(aligned_img_path)
        assert os.path.exists(deidentified__dataset_path)
        #evaluation
        index_aligned, label_aligned = model.fit(aligned_img_path)
        index_deidentified, label_deidentified  = model.fit(deidentified_img_path)
        #log the result
        f.writelines(f"{label_aligned}, {label_deidentified},{True if index_aligned == index_deidentified else False}")
        #increase the accuracy
        if index_aligned == index_deidentified: 
            succeses+=1

    f.close()
    accuracy = (succeses / samples)*100
    return output_score.add_metric("dan", "accuracy", "{:1.2f}%".format(accuracy))

if __name__ == "__main__":
    result, _ , _  =with_no_prints(main)
    print(result.build())