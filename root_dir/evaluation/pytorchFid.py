import argparse
import subprocess
import utils as util
def main():
    args = util.read_args()
    path_to_aligned_images = args.aligned_path
    path_to_deidentified_images = args.deidentified_path
    path_to_save = args.save_path
    dataset_name = util.get_dataset_name_from_path(path_to_aligned_images)
    technique_name = util.get_technique_name_from_path(path_to_deidentified_images)
    metrics_df= util.Metrics(name_evaluation="fid", 
                              name_dataset=dataset_name,
                              name_technique=technique_name,
                              name_score="dist")

    command = [
        "python", "-u", "image_quality/pytorch_fid/__main__.py", "--batch-size", "8",
        path_to_aligned_images,path_to_deidentified_images
    ]
    try:
        result = subprocess.run(
            command,
            cwd="root_dir/evaluation/",
            capture_output=True,
            text=True,
            check=True
        )
        fidscore =  result.stdout.split(" ")[-1].replace("\n","")
        metrics_df.add_score(path_to_aligned_images, path_to_deidentified_images,fidscore )
        metrics_df.save_to_csv(path_to_save)
        print(f"fid scores saved in {path_to_save}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running the script:\n{e.stderr}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    return
if __name__ == '__main__':
    main()
