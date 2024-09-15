import argparse
import subprocess
def main():
    parser = argparse.ArgumentParser(description="Evaluate FID score")
    parser.add_argument('aligned_path', type=str, help="Path to the aligned dataset directory")
    parser.add_argument('deidentified_path', type=str, help="Path to the deidentified directory")
    args = parser.parse_args()
    
    # Construir el comando
    command = [

        "python", "-u", "__main__.py", "--batch-size", "8",
        args.aligned_path, args.deidentified_path
    ]

    try:
        # Cambia el directorio de trabajo y ejecuta el comando
        result = subprocess.run(
            command,
            cwd="pytorch_fid",
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running the script:\n{e.stderr}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == '__main__':
    main()
