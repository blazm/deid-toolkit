import argparse
import os
import sys
import subprocess
import select
from colorama import Fore
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process")
    parser.add_argument('dataset_path', type=str, help="Path to the dataset directory")
    parser.add_argument('dataset_save', type=str, help="Path to the save directory")
    parser.add_argument('--dataset_filetype', type=str, default='jpg', help="Filetype of the dataset images (default: jpg)")
    parser.add_argument('--dataset_newtype', type=str, default='jpg', help="Filetype for the anonymized images (default: jpg)")
    args = parser.parse_args()
    
    if os.path.basename(args.dataset_save) != "rafd" :
        print(Fore.LIGHTRED_EX + f"For now, this method is only available for the Rafd dataset.") 
        print (f"Skipping {os.path.basename(args.dataset_save)}.", Fore.RESET)
        sys.exit()
    
    command = ("cd LeeCroft_GNN && "
               f"python -u generate_random.py {args.dataset_path} {args.dataset_save}")

    try:
            process = subprocess.Popen(command, shell=True, executable="/bin/bash", stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            
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
        