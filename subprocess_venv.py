import os
import subprocess

def create_mamba_env(env_name, env_file=None):
    """Creates a virtual environment with mamba if it doesn't exist."""
    result = subprocess.run(['mamba', 'env', 'list'], capture_output=True, text=True)
    
    if env_name not in result.stdout:
        if env_file and os.path.exists(env_file):
            print(f"Creating the virtual environment '{env_name}' from file '{env_file}' with mamba.")
            subprocess.run(['mamba', 'env', 'create', '-f', env_file, '-n', env_name])
        else:
            print(f"Creating the virtual environment '{env_name}' with mamba.")
            subprocess.run(['mamba', 'create', '-n', env_name, 'python', '-y'])
    else:
        print(f"The virtual environment '{env_name}' already exists.")

def run_script_in_mamba_env(env_name, script_path):
    """Runs a script in a virtual environment created with mamba."""
    # Ensure the virtual environment is created
    create_mamba_env(env_name)
    
    # Path to the temporary shell script
    temp_script_path = os.path.join(os.getcwd(), "run_in_mamba_env.sh")
    
    # Create the shell script content
    script_content = f"""#!/bin/bash
# Initialize conda
eval "$(conda shell.bash hook)"
# Activate the specified conda environment
conda activate {env_name}
# Run the specified Python script with unbuffered output
python -u {script_path}
"""
    
    # Write the shell script to a file
    with open(temp_script_path, 'w') as f:
        f.write(script_content)
    
    # Make the shell script executable
    os.chmod(temp_script_path, 0o755)
    
    # Execute the shell script
    process = subprocess.Popen(['/bin/bash', temp_script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Read and print the output in real-time with a delay
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    
    # Read and print errors in real-time with a delay
    stderr = process.communicate()[1]
    if stderr:
        print("Errors:", stderr.strip())
    
    # Clean up the temporary shell script
    os.remove(temp_script_path)

if __name__ == "__main__":
    # Name of the virtual environment
    env_name = 'test_env'
    # Path to the script to be run in the virtual environment
    script_to_run = 'test_script.py'
    
    run_script_in_mamba_env(env_name, script_to_run)
