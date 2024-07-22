import subprocess
import select
import os
import argparse

def main(input_path, output_path):
    dataset_name = os.path.basename(input_path)
    preprocess_output = os.path.join('ciagan', 'dataset', 'preprocessed', dataset_name)
    scripts_path = os.path.join('ciagan','source')
    if not os.path.exists(preprocess_output):
        os.makedirs(preprocess_output)
    command_process = (
        f"cd {scripts_path} && "
        f"python -u process_data.py --input {input_path} --output {preprocess_output}"
    )
    process = subprocess.Popen(command_process, shell=True, executable="/bin/bash", stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

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

    command_deid = (
        f'cd {scripts_path} && '
        f'python -u test.py --data {preprocess_output} --out {output_path}'
    )
    process = subprocess.Popen(command_deid, shell=True, executable="/bin/bash", stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process and anonymize images.")
    parser.add_argument('input_path', type=str, help='Path to input directory')
    parser.add_argument('ouput_path', type=str, help='Path to output directory')

    args = parser.parse_args()
    main(input_path=args.input_path, output_path=args.ouput_path)
