import argparse
import os
import lpips
import torch
import numpy as np
from PIL import Image
from utils import compute_mean_std,resize_if_different,get_output_filename, MetricsBuilder
def main():
    output_result = MetricsBuilder()
    parser = argparse.ArgumentParser(description="Evaluate lpips score")
    parser.add_argument('path', type=str, nargs=2,
                    help=('Paths of the datasets aligned and deidentified'))

    args = parser.parse_args()
    aligned_dataset_path = args.path[0]
    deid_dataset_path  = args.path[1] #deidentified datast
    output_scores_file = get_output_filename("lpips", aligned_dataset_path, deid_dataset_path)
    use_gpu = False
    #True if torch.cuda.is_available() else False
    
    loss_fn = lpips.LPIPS(net='alex', version="0.1") # alex
    if use_gpu:
        output_result.add_misc_message("Cuda is available")
        loss_fn.cuda()
    else: 
        output_result.add_misc_message("Cuda is not available")
    
    f = open(output_scores_file, 'w')
    files = os.listdir(aligned_dataset_path)
    for file in files:
        if(os.path.exists(os.path.join(deid_dataset_path,file))):
            # Load images
            aligned_img_path = os.path.join(aligned_dataset_path, file)
            deidentified_img_path = os.path.join(deid_dataset_path, file)
            img1 = Image.open(deidentified_img_path) #deidentified image
            img0 = resize_if_different(Image.open(aligned_img_path), img1) #aligned image
            #convert to tensors
            img0 = lpips.im2tensor(np.array(img0))  # RGB image from [-1,1]
            img1 = lpips.im2tensor(np.array(img1))  # RGB image from [-1,1]
            
            img0.cuda() if use_gpu else img0.cpu()
            img1.cuda() if use_gpu else img1.cpu()

            # Compute distance
            dist01 = loss_fn.forward(img0,img1)
            f.writelines('%.6f\n'%(dist01)) # we need only scores, to compute averages easily

    f.close()
    
    mean, std = compute_mean_std(output_scores_file)
    output_result.add_metric("lpips", "mean", "{:1.2f}".format(mean))
    output_result.add_metric("lpips", "std", "{:1.2f}".format(std))
    print(output_result.build())
if __name__ == '__main__':
    main()
