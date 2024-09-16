import argparse
import subprocess
import torch
import os
import lpips
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Evaluate FID score")
    parser.add_argument('path', type=str, nargs=2,
                    help=('Paths of the datasets aligned and deidentified'))

    args = parser.parse_args()
    aligned_dataset_path = args.path[0]
    deid_dataset_path  = args.path[1] #deidentified datast
    dataset_name = aligned_dataset_path.split("/")[-1]
    technique_name = deid_dataset_path.split("/")[-2]

    output_scores_file = f"./root/evaluation/output/lpips_{dataset_name}_{technique_name}.txt" #TODO: fix this to absolute path
    use_gpu = True if torch.cuda.is_available() else False
    
    from torch import nn
    loss_fn = lpips.LPIPS(net='alex', version="0.1") # alex
    if use_gpu:
        print("\nCUDA is available")
    else: 
        print("\nNot CUDA available")
    
    f = open(output_scores_file, 'w')
    files = os.listdir(aligned_dataset_path)
    for file in files:
        if(os.path.exists(os.path.join(deid_dataset_path,file))):
            # Load images
            img0 = lpips.im2tensor(lpips.load_image(os.path.join(aligned_dataset_path,file))) # RGB image from [-1,1]
            img1 = lpips.im2tensor(lpips.load_image(os.path.join(deid_dataset_path,file)))

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

            #print("Shapes: ", img0.shape, " ", img1.shape)
            # Compute distance
            #dist01 = loss_fn(img0, img1)
            dist01 = loss_fn.forward(img0,img1)
            
            #print('%s: %.3f'%(file,dist01)) # if using spatial, we need .mean()
            #f.writelines('%s: %.6f\n'%(file,dist01)) # original saves image name and score
            f.writelines('%.6f\n'%(dist01)) # we need only scores, to compute averages easily

    f.close()
    arr = np.loadtxt(output_scores_file)
    print("lpips | mean & std: " + "{:1.2f}".format(arr.mean()) + " ± " + "{:1.2f}".format(arr.std()))
if __name__ == '__main__':
    main()
