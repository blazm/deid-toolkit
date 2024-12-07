import argparse
import os
import lpips
import torch
import numpy as np
from PIL import Image
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import utils as util
from tqdm import tqdm

# Now you can import the functions and classes

# X: (N,3,H,W) a batch of non-negative RGB images (0~255)
# Y: (N,3,H,W)  
def main():
    args = util.read_args()
    #get the mandatory args
    #get the only two params
    aligned_dataset_path = args.aligned_path
    deidentified_path = args.deidentified_path
    path_to_save = args.save_path
    path_to_log = args.dir_to_log

    dataset_name = util.get_dataset_name_from_path(aligned_dataset_path)
    technique_name = util.get_technique_name_from_path(deidentified_path)

    ssim_df= util.Metrics(name_score="ssim")
    msssim_df= util.Metrics(name_score="msssim")
    
    #build the ouput files
    #ssim_output_scores_file = util.get_output_filename("ssim", aligned_dataset_path, deidentified_path)
    #msssim_output_scores_file = util.get_output_filename("mssim", aligned_dataset_path, deidentified_path)

    use_gpu = True if torch.cuda.is_available() else False

    from torch import nn
    loss_fn = nn.MSELoss()
    if use_gpu:
        loss_fn.cuda()
 
    
    #f = open(ssim_output_scores_file, 'w')
    #f_ms = open(msssim_output_scores_file,'w')
    files = os.listdir(aligned_dataset_path)

    for file in tqdm(files, total=len(files), desc=f"ssim | {dataset_name}-{technique_name} "):
        if(os.path.exists(os.path.join(aligned_dataset_path,file))):
            # Load images
            aligned_img_path = os.path.join(aligned_dataset_path, file)
            deidentified_img_path = os.path.join(deidentified_path, file)
            if not os.path.exists(aligned_img_path):
                util.log(os.path.join(path_to_log,"ssim.txt"), 
                        f"({dataset_name}) The source images are not in {aligned_img_path} ")
                print(f"{aligned_dataset_path} does not exist")
                continue
            if not  os.path.exists(deidentified_img_path):
                util.log(os.path.join(path_to_log,"ssim.txt"), 
                        f"({technique_name}) The deidentified images are not in {deidentified_img_path} ")
                print(f"{deidentified_img_path} does not exist")
                continue

            img1 = Image.open(deidentified_img_path)
            img0 = util.resize_if_different(Image.open(aligned_img_path), img1)
            #convert to tensor
            img0 = lpips.im2tensor(np.array(img0))  # RGB image from [-1,1]
            img1 = lpips.im2tensor(np.array(img1))  # RGB image from [-1,1]

            img0.cuda() if use_gpu else img0.cpu()
            img1.cuda() if use_gpu else img1.cpu()

            img0 = (img0 + 1.) / 2.  # [-1, 1] => [0, 1]
            img1 = (img1 + 1.) / 2. 
            ssim_val = ssim(img0, img1, data_range=1, size_average=True) # return scalar
            ms_ssim_val = ms_ssim( img0, img1, data_range=1, size_average=True ) # return scalar
            ssim_df.add_score(img=file,metric_result='%.6f'%(ssim_val))
            msssim_df.add_score(img=file, metric_result='%.6f'%(ms_ssim_val))

    ssim_df.save_to_csv(path_to_save)
    mssim_path = path_to_save.replace("ssim.csv", "msssim.csv")
    msssim_df.save_to_csv(mssim_path)
    print(f"ssim scores saved in {path_to_save}")
    print(f"mssim scores saved in {mssim_path}")

if __name__ == "__main__":
    main()