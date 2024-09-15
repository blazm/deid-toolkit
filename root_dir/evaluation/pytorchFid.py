import argparse
import subprocess
def main():
    parser = argparse.ArgumentParser(description="Evaluate FID score")
    parser.add_argument('path', type=str, nargs=2,
                    help=('Paths of the datasets aligned and deidentified'))

    args = parser.parse_args()
    print(args.path[0], args.path[1])
    # Construir el comando
    command = [
        "python", "-u", "./image_quality/__main__.py", "--batch-size", "8",
        args.path[0], args.path[1]
    ]

    try:
        # Cambia el directorio de trabajo y ejecuta el comando
        result = subprocess.run(
            command,
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
