from modules.utils.PipelineStage import IPipelineStage
import subprocess
import os

from colorama import Fore

from modules.utils import ConfigManager  

class Datasets(IPipelineStage):
    def __init__(self, stage_name):
        super().__init__(stage_name)
    def initial_update(self, aligned_folder, original_folder):
        datasets_name = ""
        # Check if the config section for datasets exists; if not, create it
        if not self.config.has_section("Available Datasets"):
            self.config.add_section("Available Datasets")
            self.config.set("Available Datasets", "original", "")
            self.config.set("Available Datasets", "aligned", "")

        # Process the original datasets
        if os.path.exists(original_folder):
            datasets = os.listdir(original_folder)
            datasets.sort()
            for dataset in datasets:
                if os.path.isdir(os.path.join(original_folder, dataset)):
                    datasets_name += (dataset + " ")
            # Remove the trailing space
            datasets_name = datasets_name.strip()
            self.config.set("Available Datasets", "original", datasets_name)
            datasets_name = ''
        else:
            print(Fore.RED + 'Original datasets directory not found. Does the ROOT_DIR ({0}) have a folder named "datasets"?'.format(self.root_dir), Fore.RESET)

        # Process the aligned datasets
        if os.path.exists(aligned_folder):
            datasets = os.listdir(aligned_folder)
            datasets.sort()
            for dataset in datasets:
                if os.path.isdir(os.path.join(aligned_folder, dataset)):
                    datasets_name += (dataset + " ")
            # Remove the trailing space
            datasets_name = datasets_name.strip()
            self.config.set("Available Datasets", "aligned", datasets_name)
            datasets_name = ''
        else:
            print(Fore.RED + 'Aligned datasets directory not found. Does the ROOT_DIR ({0}) have a folder named "datasets"?'.format(self.root_dir), Fore.RESET)

        configini_filename = ConfigManager.get_instance().filename_config_toolkit

        # Save the configuration to the file
        with open(configini_filename, "w") as configfile:
            self.config.write(configfile)        

    def do_select(self, *arg):
        "List and interactively let user select datasets: SELECT_DATASETS"
        available_datasets = self.do_list()
        aligned_datasets = self.config.get("Available Datasets", "aligned").split(" ")
        print(Fore.CYAN + "Select datasets by entering their numbers separated by space", Fore.RESET)
        # Prompt the user to select datasets
        selected_datasets_indices = input("Selection: ").split()

        # Validate and display the user's selections
        selected_datasets = []
        for i in selected_datasets_indices:
            try:
                index = int(i)
                if 0 <= index < len(available_datasets):
                    selected_datasets.append(available_datasets[index])
                    if available_datasets[index] in aligned_datasets:
                        print(Fore.LIGHTGREEN_EX + "\t" + str(index) + ". " + available_datasets[index], Fore.RESET)
                    else:
                        print(Fore.LIGHTYELLOW_EX + "\t" + str(index) + ". " + available_datasets[index], Fore.RESET)
                else:
                    print(Fore.RED + "Invalid dataset number: ", i, Fore.RESET)
            except ValueError:
                print(Fore.RED + "Invalid input, not a number: ", i, Fore.RESET)

        # Create a config section if it doesn't exist and save selected datasets
        if not self.config.has_section("selection"):
            self.config.add_section("selection")
        self.config.set("selection", "datasets", " ".join(selected_datasets))

        configini_filename = ConfigManager.get_instance().filename_config_toolkit
        # Save the configuration
        with open(configini_filename, "w") as configfile:
            self.config.write(configfile)
    def get_selection(self, *args):
        selected_datasets = self.config.get("selection","datasets").split(" ")
        return selected_datasets

    def do_list(self, *args):
        # Retrieve available datasets from the configuration
        original_datasets = self.config.get("Available Datasets", "original").split(" ")
        aligned_datasets = self.config.get("Available Datasets", "aligned").split(" ")

        # Merge available datasets into a single list
        available_datasets = set(original_datasets) | set(aligned_datasets)
        if not available_datasets:
            print("No datasets available")
            return
        available_datasets = sorted(available_datasets)

        # Check if the root directory is set
        if self.root_dir is None:
            print(Fore.RED + "Root directory not set, set it with SET_ROOT", Fore.RESET)
            return
        # Print instructions and legend for dataset selection
        print(Fore.GREEN + "Green name = aligned version available", Fore.RESET)
        print(Fore.YELLOW + "Yellow name = only non-aligned version available", Fore.RESET)
        print(Fore.LIGHTBLACK_EX + "Gray name = dataset name on display", Fore.RESET)
        print(Fore.CYAN + "[Datasets]", Fore.RESET)
        # Display available datasets with appropriate colors
        for i, dataset in enumerate(available_datasets):
            dataset_rename = self.module_settings.datasets.get(dataset, {"rename": dataset}).get("rename", dataset) 
            if dataset in aligned_datasets:
                print(Fore.LIGHTGREEN_EX + "\t" + str(i) + ". " + dataset + Fore.LIGHTBLACK_EX+  " ("+dataset_rename+")", Fore.RESET)
            else:
                print(Fore.LIGHTYELLOW_EX + "\t" + str(i) + ". " + dataset + Fore.LIGHTBLACK_EX+  " ("+dataset_rename+")", Fore.RESET)
        return available_datasets
    
    def do_run(self, *args):
        print("Method run have not been implemented yet for datasets")
        return super().do_run()
    def _list_available_datasets(self):
        path = os.path.join(self.root_dir,ConfigManager.get_instance().FOLDER_DATASET,"aligned")
        if os.path.exists(path):
            return os.listdir(path)
        else:
            print(f"Dataset directory not found: {path}")
            return []
