import argparse
import os
import lpips
import torch
import numpy as np
from PIL import Image
import utils as util
from tqdm import tqdm

def main():

    args = util.read_args()
    aligned_dataset_path = args.aligned_path
    deid_dataset_path  = args.deidentified_path
    path_to_log = args.dir_to_log

    path_to_save = args.save_path
    dataset_name = util.get_dataset_name_from_path(aligned_dataset_path)
    technique_name = util.get_technique_name_from_path(deid_dataset_path)
    metrics_df= util.Metrics(name_score="lpips")

    #output_scores_file = util.get_output_filename("lpips", aligned_dataset_path, deid_dataset_path)
    use_gpu = False
    #True if torch.cuda.is_available() else False
    
    loss_fn = lpips.LPIPS(net='alex', version="0.1") # alex
    if use_gpu:
        loss_fn.cuda()
    
    
    #f = open(output_scores_file, 'w')
    files = os.listdir(aligned_dataset_path)
    for file in tqdm(files, total=len(files),  desc=f"lpips | {dataset_name}-{technique_name}"):
        if(os.path.exists(os.path.join(deid_dataset_path,file))):
            # Load images
            aligned_img_path = os.path.join(aligned_dataset_path, file)
            deidentified_img_path = os.path.join(deid_dataset_path, file)
            if not os.path.exists(aligned_img_path):
                util.log(os.path.join(path_to_log,"lpips.txt"), 
                        f"({dataset_name}) The source images are not in {aligned_img_path} ")
                print(f"{aligned_dataset_path} does not exist")
                continue
            if not  os.path.exists(deidentified_img_path):
                util.log(os.path.join(path_to_log,"lpips.txt"), 
                        f"({technique_name}) The deidentified images are not in {deidentified_img_path} ")
                print(f"{deidentified_img_path} does not exist")
                continue
            img1 = Image.open(deidentified_img_path) #deidentified image
            img0 = util.resize_if_different(Image.open(aligned_img_path), img1) #aligned image
            #convert to tensors
            img0 = lpips.im2tensor(np.array(img0))  # RGB image from [-1,1]
            img1 = lpips.im2tensor(np.array(img1))  # RGB image from [-1,1]
            
            img0.cuda() if use_gpu else img0.cpu()
            img1.cuda() if use_gpu else img1.cpu()

            # Compute distance
            dist01 = loss_fn.forward(img0,img1)
            metrics_df.add_score(img=file,
                                 metric_result='%.6f'%(dist01))
            #f.writelines('%.6f\n'%(dist01)) # we need only scores, to compute averages easily
    #f.close()
    #mean, std = util.compute_mean_std(output_scores_file)
    metrics_df.save_to_csv(path_to_save)
    print(f"lpips saved into {path_to_save}")
    return
if __name__ == '__main__':
    main()
