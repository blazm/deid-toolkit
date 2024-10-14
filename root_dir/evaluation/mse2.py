import argparse
import os
from unittest import result
import lpips
import torch
import numpy as np
from PIL import Image
from utils import MetricsBuilder, compute_mean_std, get_output_filename, resize_if_different, with_no_prints, read_args


def main(output_result: MetricsBuilder):
    args = read_args()
    aligned_path = args.aligned_path
    deidentified_path = args.deidentified_path

    output_scores_file = get_output_filename("mse", aligned_path, deidentified_path)

    use_gpu = True if torch.cuda.is_available() else False
    from torch import nn
    loss_fn = nn.MSELoss()
    if use_gpu:
        loss_fn.cuda()
        
    f = open(output_scores_file, 'w')
    files = os.listdir(aligned_path)

    for file in files:
        aligned_img_path = os.path.join(aligned_path, file)
        deidentified_img_path = os.path.join(deidentified_path, file)
        
        if os.path.exists(deidentified_img_path): # check if the deidentified image exist
            # Load images
            img1 = Image.open(deidentified_img_path) # deidentified one
            img0 = resize_if_different(Image.open(aligned_img_path), img1) #the aligned image
            # Convert to tensors
            img0 = lpips.im2tensor(np.array(img0))  # RGB image from [-1,1]
            img1 = lpips.im2tensor(np.array(img1))  # RGB image from [-1,1]

            img0.cuda() if use_gpu else img0.cpu()
            img1.cuda() if use_gpu else img1.cpu()
            # Compute MSE distance
            dist01 = loss_fn(img0, img1)
            f.writelines('%.6f\n' % dist01)
    
    f.close()
    # Calculate mean and standard deviation
    mean, std = compute_mean_std(output_scores_file)
    output_result.add_output_message("Cuda is available" if use_gpu else "Not cuda available")
    return output_result.add_metric("mse", "mean ± std", "{:1.2f} ± {:1.2f}".format(mean, std))
if __name__ == '__main__':
    output_result = MetricsBuilder()
    try:
        result, output , _ =with_no_prints(main, output_result)
        
    except Exception as e:
        output_result.add_error(f"Unexpected error: {e}")
    #    print(output_result.build())
    print(output_result.build())
