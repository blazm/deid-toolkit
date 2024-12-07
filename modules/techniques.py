from PipelineStage import PipelineStage
from utils import ConfigManager
from colorama import Fore  # color text
import os
import subprocess
import select

class Techniques(PipelineStage):
    def __init__(self, stage_name):
        super().__init__(stage_name)
        self.config = ConfigManager.get_instance().config_toolkit
        self.module_settings = ConfigManager.get_instance().config_modules    
    def initial_update(self,techniques_folder):
        techniques_name = ""
         # Check if the config section for techniques exists; if not, create it
        if not self.config.has_section("Available Techniques"):
            self.config.add_section("Available Techniques")
            self.config.set("Available Techniques", "techniques", "")

        # Process the techniques folder
        if os.path.exists(techniques_folder):
            techniques = os.listdir(techniques_folder)
            techniques.sort()
            for technique in techniques:
                # Check if it's a python file and not a directory (pycache i s problem)
                if os.path.basename(os.path.join(techniques_folder,technique)).endswith("py"):
                    # Remove the .py extension
                    technique_name = technique[:-3]
                    techniques_name += (technique_name + " ")
            # Remove the trailing space
            techniques_name = techniques_name.strip()
            self.config.set("Available Techniques", "techniques", techniques_name)
        else:
            print(Fore.RED + 'Techniques directory not found. Does the ROOT_DIR ({0}) have a folder named "techniques"?'.format(self.root_dir), Fore.RESET)

        # Save the configuration to the file
        with open(self._config_file_name, "w") as configfile:
            self.config.write(configfile)
        pass
    def get_available(self):
        # Retrieve available techniques from the configuration
        techniques = self.config.get("Available Techniques", "techniques").split(" ")

        if not techniques:
            print("No techniques available")
            return
        techniques = sorted(techniques)
        # Check if the root directory is set
        if self.root_dir is None:
            print(Fore.RED + "Root directory not set, set it with SET_ROOT", Fore.RESET)
            return

        # Print instructions for technique selection
        print(Fore.CYAN + "[Techniques]", Fore.RESET)
        print(Fore.LIGHTBLACK_EX + "Gray name = technique name on display", Fore.RESET)
        # Display available techniques
        for i, technique in enumerate(techniques):
            techniques_rename = self.modules_settings.techniques.get(technique, {"rename": technique}).get("rename", technique) 
            print(Fore.LIGHTYELLOW_EX + "\t" + str(i) + ". " + technique+Fore.LIGHTBLACK_EX +" ("+techniques_rename+")", Fore.RESET)

        return techniques

    def select(self, arg):
        """implement this method is mandatory"""
        "List and interactively let user select techniques: SELECT_TECHNIQUES"
        available_techniques = self.get_available_techniques()

        if not available_techniques:
            print("No techniques available")
            return
        print(Fore.CYAN + "Select techniques by entering their numbers separated by space", Fore.RESET)
        # Prompt the user to select techniques
        selected_techniques_indices = input("Selection: ").split()

        # Validate and display the user's selections
        selected_techniques = []
        for i in selected_techniques_indices:
            try:
                index = int(i)
                if 0 <= index < len(available_techniques):
                    selected_techniques.append(available_techniques[index])
                    print(Fore.LIGHTYELLOW_EX + "\t" + str(index) + ". " + available_techniques[index], Fore.RESET)
                else:
                    print(Fore.RED + "Invalid technique number: ", i, Fore.RESET)
            except ValueError:
                print(Fore.RED + "Invalid input, not a number: ", i, Fore.RESET)

        # Create a config section if it doesn't exist and save selected techniques
        if not self.config.has_section("selection"):
            self.config.add_section("selection")
        self.config.set("selection", "techniques", " ".join(selected_techniques))

        # Save the configuration
        with open("config.ini", "w") as configfile:
            self.config.write(configfile)
    def run(self):
        "Run selected techniques on selected datasets: RUN_TECHNIQUES"
        print("Running techniques")
        if not self.config.has_option("selection", "techniques") or not self.config.has_option("selection", "datasets"):
            print("No datasets or techniques selected.")
            return

        selected_datasets_names = self.config.get("selection", "datasets").split()
        selected_techniques_names = self.config.get("selection", "techniques").split()

        for technique_name in selected_techniques_names:
            try:
                #TODO use this as a part of other module, no for techniques
                venv_exists = self.check_and_create_conda_env(technique_name)
                venv_name = 'toolkit'
                if venv_exists:
                    venv_name = technique_name
                for dataset_name in selected_datasets_names:
                    try:
                        self._process_dataset_with_technique(technique_name, venv_name, dataset_name)
                    except (ValueError, IndexError) as e:
                        print(f"Invalid dataset index: {dataset_name}. Error: {e}")
            except (ValueError, IndexError) as e:
                print(f"Invalid technique index: {technique_name}. Error: {e}")
        # TODO: every technique must have a python script that can be run and deidentify either a single file or a directory
        # the script should be able to take input and output directories as arguments
    def _process_dataset_with_technique(self, technique_name, venv_name, dataset_name):
        answer = ''
        original_dataset_path = os.path.join(self.root_dir, 
                                             ConfigManager.get_instance().FOLDER_DATASET,
                                             "original", dataset_name, "img")
        if technique_name is None:
            print(f"Technique module is None for {technique_name}")
            return

        aligned_dataset_path = os.path.join(self.root_dir, 
                                            ConfigManager.get_instance().FOLDER_DATASET,
                                            "aligned", dataset_name)
        if not os.path.exists(aligned_dataset_path):
            print(f"{dataset_name} dataset has not been preprocessed yet. Do you want to preprocess it first? [y/n]")
            answer = input("Answer: ")
            if answer == 'y':
                import align_face_mtcnn
                align_face_mtcnn.main(dataset_name=dataset_name, dataset_path=original_dataset_path,
                                    dataset_save_path=aligned_dataset_path)
            else:
                print(f"Running {technique_name} on unaligned {dataset_name} dataset")
                aligned_dataset_path = original_dataset_path

        dataset_save_path = os.path.join(self.root_dir, 'datasets', technique_name, dataset_name)

        if not os.path.exists(dataset_save_path):
            os.makedirs(dataset_save_path)

        print(f"Processing dataset: {dataset_name} | Source path: {aligned_dataset_path} | Save path: {dataset_save_path}")

        self.run_technique_script(venv_name, technique_name, aligned_dataset_path, dataset_save_path)
    def run_technique_script(self, venv_name, technique_name, aligned_dataset_path, dataset_save_path):
        conda_sh_path = os.path.expanduser(ConfigManager.get_instance().CONDA_DOT_SH_PATH)

        if not os.path.exists(conda_sh_path):
            print("conda.sh path does'nt exist, please change it in run_script() in deid_toolkit.py")

        aligned_dataset_path = os.path.abspath(aligned_dataset_path)
        dataset_save_path = os.path.abspath(dataset_save_path)
        path_technique_folder = os.path.join(
            self.root_dir,
            ConfigManager.get_instance().FOLDER_TECHNIQUES)

        command = (
            f"source {conda_sh_path} && "
            f"conda activate {venv_name} && "
            f"cd {path_technique_folder} && "
            f"python -u {technique_name}.py {aligned_dataset_path} {dataset_save_path} "
        )
        
        try:
            process = subprocess.Popen(command, shell=True,
                                        executable="/bin/bash",
                                        stdout=subprocess.PIPE, 
                                        stderr=subprocess.PIPE, text=True)
            
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
        