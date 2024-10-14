import argparse
import os
import lpips
import torch
import numpy as np
from PIL import Image
import utils as util

def main():
    output_result = util.MetricsBuilder()
    parser = argparse.ArgumentParser(description="Evaluate lpips score")
    parser.add_argument('path', type=str, nargs=2,
                    help=('Paths of the datasets aligned and deidentified'))

    args = util.read_args()
    aligned_dataset_path = args.aligned_path
    deid_dataset_path  = args.deidentified_path
    path_to_save = args.save_path
    dataset_name = util.get_dataset_name_from_path(aligned_dataset_path)
    technique_name = util.get_technique_name_from_path(deid_dataset_path)
    metrics_df= util.Metrics(name_evaluation="lpips", 
                              name_dataset=dataset_name,
                              name_technique=technique_name,
                              name_score="dist")

    output_scores_file = util.get_output_filename("lpips", aligned_dataset_path, deid_dataset_path)
    use_gpu = False
    #True if torch.cuda.is_available() else False
    
    loss_fn = lpips.LPIPS(net='alex', version="0.1") # alex
    if use_gpu:
        loss_fn.cuda()
    
    
    f = open(output_scores_file, 'w')
    files = os.listdir(aligned_dataset_path)
    for file in files:
        if(os.path.exists(os.path.join(deid_dataset_path,file))):
            # Load images
            aligned_img_path = os.path.join(aligned_dataset_path, file)
            deidentified_img_path = os.path.join(deid_dataset_path, file)
            img1 = Image.open(deidentified_img_path) #deidentified image
            img0 = util.resize_if_different(Image.open(aligned_img_path), img1) #aligned image
            #convert to tensors
            img0 = lpips.im2tensor(np.array(img0))  # RGB image from [-1,1]
            img1 = lpips.im2tensor(np.array(img1))  # RGB image from [-1,1]
            
            img0.cuda() if use_gpu else img0.cpu()
            img1.cuda() if use_gpu else img1.cpu()

            # Compute distance
            dist01 = loss_fn.forward(img0,img1)
            metrics_df.add_score(path_aligned=aligned_img_path,
                                 path_deidentified=deidentified_img_path,
                                 metric_result=dist01)
            f.writelines('%.6f\n'%(dist01)) # we need only scores, to compute averages easily

    f.close()
    output_result.add_output_message("Executed on Cuda" if use_gpu else "Executed on cpu")
    mean, std = util.compute_mean_std(output_scores_file)
    output_result.add_metric("lpips", "mean ± std", "{:1.2f} ± {:1.2f}".format(mean, std))
    metrics_df.save_to_csv(path_to_save)
    return output_result.build()
if __name__ == '__main__':
    result, output , _ =util.with_no_prints(main)
    print(result)
