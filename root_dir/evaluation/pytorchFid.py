import argparse
import subprocess
def main():
    parser = argparse.ArgumentParser(description="Evaluate FID score")
    parser.add_argument('path', type=str, nargs=2,
                    help=('Paths of the datasets aligned and deidentified'))

    args = parser.parse_args()
    command = [
        "python", "-u", "image_quality/pytorch_fid/__main__.py", "--batch-size", "8",
        args.path[0], args.path[1]
    ]

    try:
        result = subprocess.run(
            command,
            cwd="root_dir/evaluation/",
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout.split(" ")[-1])
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running the script:\n{e.stderr}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == '__main__':
    main()
