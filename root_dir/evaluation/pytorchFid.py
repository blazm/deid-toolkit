import argparse
import subprocess
import utils as util
def main():
    output_result = util.MetricsBuilder()
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
        output_result.add_metric("pytorchFid", "score",fidscore)
        metrics_df.add_score(path_to_aligned_images, path_to_deidentified_images,fidscore )
        metrics_df.save_to_csv(path_to_save)
    except subprocess.CalledProcessError as e:
        output_result.add_error(f"Error occurred while running the script:\n{e.stderr}")
    except Exception as e:
        output_result.add_error(f"Unexpected error: {e}")
    return output_result.build()
if __name__ == '__main__':
    result, _, _ =util.with_no_prints(main)
    print(result)
