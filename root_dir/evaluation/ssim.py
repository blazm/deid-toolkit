import argparse
import os
import lpips
import torch
import numpy as np
from PIL import Image


from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

def resize(img0, img1):
    # Resize img0 to match img1's size
    reference_size = img1.size
    return img0.resize(reference_size)
# Now you can import the functions and classes

# X: (N,3,H,W) a batch of non-negative RGB images (0~255)
# Y: (N,3,H,W)  
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = argparse.ArgumentParser(description="Evaluate ssim score between aligned and deidentified images")
    parser.add_argument('path', type=str, nargs=2,
                        help=('Paths of the aligned and deidentified datasets'))
    args = parser.parse_args()
    aligned_dataset_path = args.path[0]
    deidentified_path = args.path[1]

    dataset_name = aligned_dataset_path.split("/")[-1]
    technique_name = deidentified_path.split("/")[-2]

    ssim_output_scores_file = f"./root_dir/evaluation/output/ssim_{dataset_name}_{technique_name}.txt" #TODO: fix this to absolute path
    msssim_output_scores_file = f"./root_dir/evaluation/output/msssim_{dataset_name}_{technique_name}.txt"
    use_gpu = True if torch.cuda.is_available() else False

    from torch import nn
    loss_fn = nn.MSELoss()
    if use_gpu:
        print("\nCUDA is available\n")
        loss_fn.cuda()
    else: 
        print("\nNot CUDA available\n")
    
    f = open(ssim_output_scores_file, 'w')
    f_ms = open(msssim_output_scores_file,'w')
    files = os.listdir(aligned_dataset_path)

    for file in files:
        if(os.path.exists(os.path.join(aligned_dataset_path,file))):
            # Load images
            aligned_img_path = os.path.join(aligned_dataset_path, file)
            deidentified_img_path = os.path.join(deidentified_path, file)
            img0 = Image.open(aligned_img_path)
            img1 = Image.open(deidentified_img_path)

            if img0.size != img1.size:
                img0 = resize(img0, img1)

            img0 = lpips.im2tensor(np.array(img0))  # RGB image from [-1,1]
            img1 = lpips.im2tensor(np.array(img1))  # RGB image from [-1,1]
             #if img1.shape[2] == 128: # CIAGAN images need to be resized
            #	#from torchvision import transforms
            #	#trans = transforms.Compose([transforms.Resize(1024, 1024)])
            #	img1 = lpips.upsample(img1, (1024, 1024))

            if use_gpu:
                img0 = img0.cuda()
                img1 = img1.cuda()
            else: 
                img0 = img0.cpu()
                img1 = img1.cpu()
            img0 = (img0 + 1.) / 2.  # [-1, 1] => [0, 1]
            img1 = (img1 + 1.) / 2. 
            ssim_val = ssim( img0, img1, data_range=1, size_average=True) # return scalar
            ms_ssim_val = ms_ssim( img0, img1, data_range=1, size_average=True ) # return scalar


            #print('%s: %.3f'%(file,dist01)) # if using spatial, we need .mean()
            #f.writelines('%s: %.6f\n'%(file,dist01)) # original saves image name and score
            f.writelines('%.6f\n'%(ssim_val)) # we need only scores, to compute averages easily
            f_ms.writelines('%.6f\n'%(ms_ssim_val)) # we need only scores, to compute averages easily

    f.close()
    f_ms.close()
    ssim_arr = np.loadtxt(ssim_output_scores_file)
    mssim_arr= np.loadtxt(msssim_output_scores_file)
    print("ssim | mean & std: {:1.2f}".format(ssim_arr.mean()) + " ± "+"{:1.2f}".format(ssim_arr.std()))
    print("msssim | mean & std: {:1.2f}".format(mssim_arr.mean()) + " ± "+"{:1.2f}".format(mssim_arr.std()))
    



if __name__ == "__main__":
    main()