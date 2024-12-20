import argparse
import os
from unittest import result
import lpips
import torch
import numpy as np
from PIL import Image
import utils as util
from tqdm import tqdm

def im2tensor(image):
    # Convert a NumPy image array to a PyTorch tensor scaled to [-1, 1]
    tensor = torch.from_numpy(image).permute(2, 0, 1).float()  # Change shape to (C, H, W)
    tensor = tensor / 127.5 - 1.0  # Scale to [-1, 1]
    return tensor
def main():
    args = util.read_args()
    
    aligned_path = args.aligned_path
    deidentified_path = args.deidentified_path
    path_to_save = args.save_path
    path_to_log = args.dir_to_log

    dataset_name = args.dataset_name #util.get_dataset_name_from_path(aligned_path)
    technique_name = args.technique_name #util.get_technique_name_from_path(deidentified_path)
    metrics_df= util.Metrics( name_score="mse")

    use_gpu = True if torch.cuda.is_available() else False
    from torch import nn
    loss_fn = nn.MSELoss()
    if use_gpu:
        loss_fn.cuda()
        
    #f = open(output_scores_file, 'w')
    files = os.listdir(aligned_path)

    for file in tqdm(files, total=len(files),  desc=f"mse | {dataset_name}-{technique_name}"):
        aligned_img_path = os.path.join(aligned_path, file)
        deidentified_img_path = os.path.join(deidentified_path, file)
        
        if os.path.exists(deidentified_img_path): # check if the deidentified image exist
            # Load images
            if not os.path.exists(aligned_img_path):
                util.log(os.path.join(path_to_log,"mse.txt"), 
                        f"({dataset_name}) The source images are not in {aligned_img_path} ")
                print(f"{aligned_img_path} does not exist")
                continue
            if not  os.path.exists(deidentified_img_path):
                util.log(os.path.join(path_to_log,"mse.txt"), 
                        f"({technique_name}) The deidentified images are not in {deidentified_img_path} ")
                print(f"{deidentified_img_path} does not exist")
                continue
            img1 = Image.open(deidentified_img_path) # deidentified one
            img0 = util.resize_if_different(Image.open(aligned_img_path), img1) #the aligned image
            # Convert to tensors
            img0 = im2tensor(np.array(img0))  # RGB image from [-1,1]
            img1 = im2tensor(np.array(img1))  # RGB image from [-1,1]

            img0.cuda() if use_gpu else img0.cpu()
            img1.cuda() if use_gpu else img1.cpu()
            # Compute MSE distance
            dist01 = loss_fn(img0, img1)
            metrics_df.add_score(img=file,
                                 metric_result='%.6f' % dist01)
            
            #f.writelines('%.6f\n' % dist01)
    
    #f.close()
    # Calculate mean and standard deviation
    #mean, std = util.compute_mean_std(output_scores_file)
    metrics_df.save_to_csv(path_to_save)
    print(f"mse scores saved in {path_to_save}")
    return 
if __name__ == '__main__':
    main()
        