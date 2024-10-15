import argparse
import os
from unittest import result
import lpips
import torch
import numpy as np
from PIL import Image
import utils as util


def main():
    args = util.read_args()
    aligned_path = args.aligned_path
    deidentified_path = args.deidentified_path
    path_to_save = args.save_path
    dataset_name = util.get_dataset_name_from_path(aligned_path)
    technique_name = util.get_technique_name_from_path(deidentified_path)
    metrics_df= util.Metrics(name_evaluation="mse", 
                              name_dataset=dataset_name,
                              name_technique=technique_name,
                              name_score="dist")

    output_scores_file = util.get_output_filename("mse", aligned_path, deidentified_path)

    use_gpu = True if torch.cuda.is_available() else False
    from torch import nn
    loss_fn = nn.MSELoss()
    if use_gpu:
        loss_fn.cuda()
        
    #f = open(output_scores_file, 'w')
    files = os.listdir(aligned_path)

    for file in files:
        aligned_img_path = os.path.join(aligned_path, file)
        deidentified_img_path = os.path.join(deidentified_path, file)
        
        if os.path.exists(deidentified_img_path): # check if the deidentified image exist
            # Load images
            img1 = Image.open(deidentified_img_path) # deidentified one
            img0 = util.resize_if_different(Image.open(aligned_img_path), img1) #the aligned image
            # Convert to tensors
            img0 = lpips.im2tensor(np.array(img0))  # RGB image from [-1,1]
            img1 = lpips.im2tensor(np.array(img1))  # RGB image from [-1,1]

            img0.cuda() if use_gpu else img0.cpu()
            img1.cuda() if use_gpu else img1.cpu()
            # Compute MSE distance
            dist01 = loss_fn(img0, img1)
            metrics_df.add_score(path_aligned=aligned_img_path,
                                 path_deidentified=deidentified_img_path, 
                                 metric_result='%.6f' % dist01)
            
            #f.writelines('%.6f\n' % dist01)
    
    #f.close()
    # Calculate mean and standard deviation
    #mean, std = util.compute_mean_std(output_scores_file)
    metrics_df.save_to_csv(path_to_save)
    print(f"mse scores save in {path_to_save}")
    return 
if __name__ == '__main__':
    main()
        