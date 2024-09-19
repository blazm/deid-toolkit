import argparse
import os
import lpips
import torch
import numpy as np
from PIL import Image
from utils import compute_mean_std, get_output_filename, resize_if_different


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = argparse.ArgumentParser(description="Evaluate MSE score between aligned and deidentified images")
    parser.add_argument('path', type=str, nargs=2,
                        help=('Paths of the aligned and deidentified datasets'))
    args = parser.parse_args()
    aligned_path = args.path[0]
    deidentified_path = args.path[1]

    output_scores_file = get_output_filename("mse", aligned_path, deidentified_path)

    use_gpu = True if torch.cuda.is_available() else False
    from torch import nn
    loss_fn = nn.MSELoss()
    if use_gpu:
        print("CUDA is available")
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
    print(" mean & std: " + "{:1.2f}".format(mean) + " ± " + "{:1.2f}".format(std))
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Unexpected error: {e}")
