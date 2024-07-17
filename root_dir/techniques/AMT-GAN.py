import argparse
import subprocess
import select

def main(dataset_path, dataset_save):
    command = (f"cd AMT-GAN && "
               f"python -u test.py --source_dir {dataset_path} --save_path {dataset_save}")
    try:
        process = subprocess.Popen(command, shell=True, executable="/bin/bash", stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        while True:
            reads = [process.stdout.fileno(), process.stderr.fileno()]
            ret = select.select(reads, [], [])
            for fd in ret[0]:
                if fd == process.stdout.fileno():
                    output = process.stdout.readline()
                    if output:
                        print(output.strip())
                if fd == process.stderr.fileno():
                    error_output = process.stderr.readline()
                    if error_output:
                        print(error_output.strip())
            
            if process.poll() is not None:
                break
        
        process.stdout.close()
        process.stderr.close()
        process.wait()

        if process.returncode != 0:
            print(f"Script exited with return code {process.returncode}")

    except Exception as e:
        print(f"Error occurred while running the script: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process and anonymize images.")
    parser.add_argument('dataset_path', type=str, help="Path to the dataset directory")
    parser.add_argument('dataset_save', type=str, help="Path to the save directory")

    args = parser.parse_args()
    main(args.dataset_path, args.dataset_save)