import argparse
import subprocess
from unittest import result
from utils import MetricsBuilder, with_no_prints, read_args
def main():
    output_result = MetricsBuilder()
    args = read_args()
    path_to_aligned_images = args.aligned_path
    path_to_deidentified_images = args.deidentified_path

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
        output_result.add_metric("pytorchFid", "score", result.stdout.split(" ")[-1].replace("\n",""))
    except subprocess.CalledProcessError as e:
        output_result.add_error(f"Error occurred while running the script:\n{e.stderr}")
    except Exception as e:
        output_result.add_error(f"Unexpected error: {e}")
    return output_result.build()
if __name__ == '__main__':
    result, _, _ =with_no_prints(main)
    print(result)
