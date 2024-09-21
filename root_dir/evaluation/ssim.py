import argparse
import os
import lpips
import torch
import numpy as np
from PIL import Image
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from utils import compute_mean_std, get_output_filename,resize_if_different, MetricsBuilder, with_no_prints
# Now you can import the functions and classes

# X: (N,3,H,W) a batch of non-negative RGB images (0~255)
# Y: (N,3,H,W)  
def main():
    output_result = MetricsBuilder()
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = argparse.ArgumentParser(description="Evaluate ssim score between aligned and deidentified images")
    parser.add_argument('path', type=str, nargs=2,
                        help=('Paths of the aligned and deidentified datasets'))
    args = parser.parse_args()
    #get the only two params
    aligned_dataset_path = args.path[0]
    deidentified_path = args.path[1]
    #build the ouput files
    ssim_output_scores_file = get_output_filename("ssim", aligned_dataset_path, deidentified_path)
    msssim_output_scores_file = get_output_filename("mssim", aligned_dataset_path, deidentified_path)

    use_gpu = True if torch.cuda.is_available() else False

    from torch import nn
    loss_fn = nn.MSELoss()
    if use_gpu:
        loss_fn.cuda()
 
    
    f = open(ssim_output_scores_file, 'w')
    f_ms = open(msssim_output_scores_file,'w')
    files = os.listdir(aligned_dataset_path)

    for file in files:
        if(os.path.exists(os.path.join(aligned_dataset_path,file))):
            # Load images
            aligned_img_path = os.path.join(aligned_dataset_path, file)
            deidentified_img_path = os.path.join(deidentified_path, file)
            img1 = Image.open(deidentified_img_path)
            img0 = resize_if_different(Image.open(aligned_img_path), img1)
            #convert to tensor
            img0 = lpips.im2tensor(np.array(img0))  # RGB image from [-1,1]
            img1 = lpips.im2tensor(np.array(img1))  # RGB image from [-1,1]

            img0.cuda() if use_gpu else img0.cpu()
            img1.cuda() if use_gpu else img1.cpu()

            img0 = (img0 + 1.) / 2.  # [-1, 1] => [0, 1]
            img1 = (img1 + 1.) / 2. 
            ssim_val = ssim(img0, img1, data_range=1, size_average=True) # return scalar
            ms_ssim_val = ms_ssim( img0, img1, data_range=1, size_average=True ) # return scalar


            #print('%s: %.3f'%(file,dist01)) # if using spatial, we need .mean()
            #f.writelines('%s: %.6f\n'%(file,dist01)) # original saves image name and score
            f.writelines('%.6f\n'%(ssim_val)) # we need only scores, to compute averages easily
            f_ms.writelines('%.6f\n'%(ms_ssim_val)) # we need only scores, to compute averages easily

    f.close()
    f_ms.close()
    ssim_mean, ssim_std = compute_mean_std(ssim_output_scores_file)
    mssim_mean, mssim_std = compute_mean_std(msssim_output_scores_file)
   
    print(output_result
          .add_metric("ssim", "mean ± std", "{:1.2f} ± {:1.2f}".format(ssim_mean, ssim_std))
          .add_metric("mssim", "mean ± std", "{:1.2f} ± {:1.2f}".format(mssim_mean, mssim_std))
          .add_output_message("Executed on cuda" if use_gpu else "Cuda not available, executed in cpu")
          .build() # don't forget to build at the end
    )

if __name__ == "__main__":
    _, result, _ = with_no_prints(main)
    print(result)