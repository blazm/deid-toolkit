import argparse
import os
import lpips
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

def resize(img0, img1):
    # Resize img0 to match img1's size
    reference_size = img1.size
    return img0.resize(reference_size)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = argparse.ArgumentParser(description="Evaluate MSE score between aligned and deidentified images")
    parser.add_argument('path', type=str, nargs=2,
                        help=('Paths of the aligned and deidentified datasets'))
    args = parser.parse_args()
    aligned_path = args.path[0]
    deidentified_path = args.path[1]

    dataset_name = aligned_path.split("/")[-1]
    technique_name = deidentified_path.split("/")[-2]

    output_scores_file = f"./root_dir/evaluation/output/mse_{dataset_name}_{technique_name}.txt" #TODO: fix this to absolute path
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
        
        if os.path.exists(deidentified_img_path):
            # Load images
            img0 = Image.open(aligned_img_path)
            img1 = Image.open(deidentified_img_path)
            
            # Resize img0 if needed
            if img0.size != img1.size:
                img0 = resize(img0, img1)
            
            # Convert to tensors
            img0 = lpips.im2tensor(np.array(img0))  # RGB image from [-1,1]
            img1 = lpips.im2tensor(np.array(img1))  # RGB image from [-1,1]
            
            if use_gpu:
                img0 = img0.cuda()
                img1 = img1.cuda()
            else:
                img0 = img0.cpu()
                img1 = img1.cpu()

            # Compute MSE distance
            dist01 = loss_fn(img0, img1)
            f.writelines('%.6f\n' % dist01)
    
    f.close()
    # Calculate mean and standard deviation
    arr = np.loadtxt(output_scores_file)
    print(" mean & std: " + "{:1.2f}".format(arr.mean()) + " ± " + "{:1.2f}".format(arr.std()))

if __name__ == '__main__':
    main()
