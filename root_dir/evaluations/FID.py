import argparse
import utils as util
import os
from pytorch_fid import fid_score
import torch

def main():
    args = util.read_args()
    path_to_aligned_images = args.aligned_path
    path_to_deidentified_images = args.deidentified_path
    path_to_save = args.save_path

    metrics_df = util.Metrics(name_score="fidscore")
    path_to_aligned_images = os.path.abspath(path_to_aligned_images)
    path_to_deidentified_images = os.path.abspath(path_to_deidentified_images)

    try:
        fidscore = fid_score.calculate_fid_given_paths(
            [path_to_aligned_images, path_to_deidentified_images],
            batch_size=8,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            dims=2048  # InceptionV3 feature layer
        )
        
        metrics_df.add_score(path_to_aligned_images, fidscore)
        metrics_df.save_to_csv(path_to_save)
        print(f"FID scores saved in {path_to_save}")

    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == '__main__':
    main()
