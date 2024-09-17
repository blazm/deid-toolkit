import sys
import argparse
import os
import lpips
import torch

module_path = os.path.join(os.path.dirname(__file__), 'image_quality', 'pytorch_ssim')
sys.path.append(module_path)
from pytorch_ssim import ssim, ms_ssim, SSIM, MS_SSIM

# Now you can import the functions and classes

# X: (N,3,H,W) a batch of non-negative RGB images (0~255)
# Y: (N,3,H,W)  
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = argparse.ArgumentParser(description="Evaluate MSE score between aligned and deidentified images")
    parser.add_argument('path', type=str, nargs=2,
                        help=('Paths of the aligned and deidentified datasets'))
    args = parser.parse_args()
    aligned_dataset_path = args.path[0]
    deidentified_path = args.path[1]

    dataset_name = aligned_dataset_path.split("/")[-1]
    technique_name = deidentified_path.split("/")[-2]

    output_scores_file = f"./root_dir/evaluation/output/ssim_{dataset_name}_{technique_name}.txt" #TODO: fix this to absolute path
    use_gpu = True if torch.cuda.is_available() else False

    from torch import nn
    loss_fn = nn.MSELoss()
    if use_gpu:
        print("\nCUDA is available")
        loss_fn.cuda()
    else: 
        print("\nNot CUDA available")
    
    f = open(output_scores_file, 'w')
    f_ms = open('MS'+output_scores_file,'w')
    files = os.listdir(aligned_dataset_path)

    for file in files:
        if(os.path.exists(os.path.join(aligned_dataset_path,file))):
            # Load images
            img0 = lpips.im2tensor(lpips.load_image(os.path.join(aligned_dataset_path,file))) # RGB image from [-1,1]
            img1 = lpips.im2tensor(lpips.load_image(os.path.join(deidentified_path,file)))
             #if img1.shape[2] == 128: # CIAGAN images need to be resized
            #	#from torchvision import transforms
            #	#trans = transforms.Compose([transforms.Resize(1024, 1024)])
            #	img1 = lpips.upsample(img1, (1024, 1024))
            #TODO: check if the images needs to be reized
            #print(img0.min(), img0.max(), img1.min(), img1.max())

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
    



if __name__ == "__main__":
    main()