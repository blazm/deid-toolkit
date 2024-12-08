from modules.utils.PipelineStage import IPipelineStage
import subprocess
import os
import select 
from colorama import Fore  # color text
from modules.utils import ConfigManager  
from modules.environments import Environments


class Evaluations(IPipelineStage):
    def __init__(self, stage_name):
        super().__init__(stage_name)
        self.__CONDA_DOT_SH_PATH = ConfigManager.get_instance().CONDA_DOT_SH_PATH
        self.__FOLDER_DATASET = ConfigManager.get_instance().FOLDER_DATASET
        self.__FOLDER_EVALUATION = ConfigManager.get_instance().FOLDER_EVALUATION
        self.__FOLDER_RESULTS = ConfigManager.get_instance().FOLDER_RESULTS
    
    def initial_update(self, evaluation_folder):
        # Check if the config section for techniques exists; if not, create it
        evaluation_names = ""
        if not self.config.has_section("Available Evaluations"):
            self.config.add_section("Available Evaluations")
            self.config.set("Available Evaluations", "evaluations", "")

        # Process the evaluation folder
        if os.path.exists(evaluation_folder):
            evaluations = os.listdir(evaluation_folder)
            evaluations.sort()
            for evaluation in evaluations:
                # Check if it's a file and not a directory
                if (os.path.isfile(os.path.join(evaluation_folder,evaluation)) 
                    and (evaluation.endswith(".py") or evaluation.endswith(".sh") )):
                        # Remove the extension
                        evaluation_name = evaluation[:-3]
                        evaluation_names += (evaluation_name + " ")
            # Remove the trailing space
            evaluation_names = evaluation_names.strip()
            self.config.set("Available Evaluations", "evaluations", evaluation_names)
        else:
            print(Fore.RED + 'Evaluation directory not found. Does the ROOT_DIR ({0}) have a folder named "evaluation"?'.format(self.root_dir), Fore.RESET)
        
        configini_filename = ConfigManager.get_instance().filename_config_toolkit
        # Save the configuration to the file
        with open(configini_filename, "w") as configfile:
            self.config.write(configfile)
    def do_list(self, *arg): 
        # Retrieve available techniques from the configuration
        evaluations = self.config.get("Available Evaluations", "evaluations").split(" ")

        if not evaluations:
            print("No evaluation method available")
            return
        evaluations = sorted(evaluations)

        # Check if the root directory is set
        if self.root_dir is None:
            print(Fore.RED + "Root directory not set, set it with SET_ROOT", Fore.RESET)
            return

        # Print instructions for technique selection
        print(Fore.CYAN + "[Evaluation methods]", Fore.RESET)
        print(Fore.LIGHTBLACK_EX + "Gray name = evaluation name on display", Fore.RESET)
        # Display available evaluation
        for i, evaluation in enumerate(evaluations):
            evaluation_rename = self.module_settings.evaluations.get(evaluation, {"rename": evaluation}).get("rename", evaluation) 

            print(Fore.LIGHTYELLOW_EX + "\t" + str(i) + ". " + evaluation+ Fore.LIGHTBLACK_EX +" ("+evaluation_rename+") ", Fore.RESET)

        return evaluations
    def do_select(self, *arg):
        "List and interactively let user select evaluation modes:  SELECT_EVALUATION"
        available_evaluations = self.do_list()

        if not available_evaluations:
            print("No evaluation method available")
            return
        print(Fore.CYAN + "Select evaluation by entering their numbers separated by space", Fore.RESET)
        # Prompt the user to select techniques
        selected_evaluations_indices =list(set(input("Selection: ").split()))

        # Validate and display the user's selections
        selected_evaluations = []
        for i in selected_evaluations_indices:
            try:
                index = int(i)
                if 0 <= index < len(available_evaluations):
                    selected_evaluations.append(available_evaluations[index])
                    print(Fore.LIGHTYELLOW_EX + "\t" + str(index) + ". " + available_evaluations[index], Fore.RESET)
                else:
                    print(Fore.RED + "Invalid technique number: ", i, Fore.RESET)
            except ValueError:
                print(Fore.RED + "Invalid input, not a number: ", i, Fore.RESET)
        # Create a config section if it doesn't exist and save selected techniques
        if not self.config.has_section("selection"):
            self.config.add_section("selection")
        self.config.set("selection", "evaluation", " ".join(selected_evaluations))

        configini_filename = ConfigManager.get_instance().filename_config_toolkit
        # Save the configuration
        with open(configini_filename, "w") as configfile:
            self.config.write(configfile)
    def get_selection(self, *args):
        selected_evaluation = self.config.get("selection","evaluation").split(" ")
        return selected_evaluation
    def do_run(self, *args):
        "Run evaluation:  RUN_EVALUATION"
        #Check if datasets or evaluation methods are selected
        if not self.config.has_option("selection", "evaluation") or not self.config.has_option("selection", "datasets"):
            print("No datasets or evaluation selected.")
            return
        #get all the selected items
        selected_datasets_names = self.config.get("selection", "datasets").split()
        selected_techniques_names = self.config.get("selection", "techniques").split()
        selected_evaluation_names = self.config.get("selection", "evaluation").split()
        if not selected_evaluation_names:
            print("No evaluation methods are selected")
            return
        #start the pipeline for each evaluation
        for evaluation_name in selected_evaluation_names:
            print(f"{evaluation_name} in progress...")
            try:
                venv_name = self.config.get("Available Environments",evaluation_name, fallback=evaluation_name)
                venv_exists =Environments.isEnvironmentExist(venv_name)
                if not venv_exists:
                    venv_name = "toolkit"
                for dataset_name in selected_datasets_names:
                    try:
                        
                        self._process_dataset_with_evaluation(evaluation_name=evaluation_name,
                                                            venv_name=venv_name,
                                                            dataset_name=dataset_name,
                                                            techniques_names=selected_techniques_names)
                    except (ValueError, IndexError) as e: 
                        print(f"Invalid dataset index: {dataset_name}. Error: {e}")
            except (ValueError, IndexError) as e:
                print(f"Invalid evaluation method index: {evaluation_name}. Error: {e}")
        return 
        #rows, headers = self._build_table_for_metrics(scores_per_evaluation)
        #TODO: print(tabulate(rows, headers=headers, tablefmt="grid"))
        # TODO: every evaluation step must have a python script that can be run and preprocess either a single file or a directory
        # the script should be able to take input and output directories as arguments
    def _process_dataset_with_evaluation(self, evaluation_name:str, venv_name:str,dataset_name:str, techniques_names:list)->list:
        """This function performs the evaluation method for one dataset (provided in params),
          with the several deidentified methods. If not file exist for the technique, skip to the next one.
          run_evaluation
        Args:
            dataset_name (str): the dataset name to evaluate
            techniques (list): techniques which build the path to access to deidentified folder
            select_metrics (list): selected metrics to evaluate
        """
        pairs_paths =[] # impostor, genuine 
        #prepare the absolutes paths for the files to evaluate
        aligned_dataset_path = os.path.join(self.root_dir, self.__FOLDER_DATASET,"aligned", dataset_name)
        aligned_dataset_path = os.path.abspath(aligned_dataset_path)
        deidentified_paths = [os.path.abspath(os.path.join(self.root_dir, 'datasets', technique, dataset_name)) 
                              for technique in techniques_names] #convert techniques into absolute deidentified paths, needed by the called function from python files
        impostor_pairs_file = os.path.abspath(os.path.join(self.root_dir, self.__FOLDER_DATASET, "pairs", f"{dataset_name}_impostor_pairs.txt"))
        genuine_pairs_file = os.path.abspath(os.path.join(self.root_dir, self.__FOLDER_DATASET, "pairs", f"{dataset_name}_genuine_pairs.txt"))
        if  os.path.exists(impostor_pairs_file) and os.path.exists(genuine_pairs_file):
            pairs_paths = (impostor_pairs_file, genuine_pairs_file)
        else:
            issue_path= os.path.join(self.root_dir, self.__FOLDER_DATASET, "pairs")
            print(f"{Fore.LIGHTRED_EX}No genuine and impostor pairs exist in {issue_path} for {dataset_name}{Fore.RESET}")
            return 

        path_evaluation  = os.path.join(self.root_dir, self.__FOLDER_EVALUATION,f"{evaluation_name}.py" ) #file to call
        path_evaluation = os.path.abspath(path_evaluation)

        print(f"Evaluation: {Fore.LIGHTCYAN_EX}{evaluation_name} -> {dataset_name} {Fore.RESET}")
        for i,deidpath_abspath in enumerate(deidentified_paths): 
            
            print(f"{Fore.GREEN}Running: {Fore.LIGHTMAGENTA_EX}{evaluation_name}{Fore.RESET} for {Fore.LIGHTMAGENTA_EX}{dataset_name}{Fore.RESET} with technique {Fore.LIGHTMAGENTA_EX}{techniques_names[i]}{Fore.RESET}...")
            if not (os.path.isdir(deidpath_abspath)): #skip if cannot find the identified dataset path
                print(f"\t>{techniques_names[i]}: Cannot find deidentified folder for {techniques_names[i]}/{dataset_name} in datasets - {Fore.LIGHTYELLOW_EX}(Skipped){Fore.RESET}")
                continue
            #get the initinal time
            save_path =os.path.abspath(os.path.join(self.root_dir,self.__FOLDER_RESULTS, f"{evaluation_name}_{dataset_name}_{techniques_names[i]}.csv"))
            #Executes the function
            self.run_evaluation_script(venv_name=venv_name, 
                                        path_evaluation=path_evaluation, 
                                        aligned_dataset_path=aligned_dataset_path,
                                        deidentified_dataset_path=deidpath_abspath,
                                        pairs = pairs_paths,
                                        save_path=save_path)
            # log the results
            #log_file = os.path.join(self.root_dir,self.logs_dir, FOLDER_EVALUATION, "evaluation_results.txt")
            #with open(log_file, 'a') as log:
            #    hours, remainder = divmod(duration, 3600)  # 3600 seconds in one hour
            #    minutes, seconds = divmod(remainder, 60)  # 60 seconds in one minute
            #    log.write("=" * 40 + "\n")
            #    log.write(f"Evaluation metric: {evaluation_name}")
            #    log.write(f" Dataset: {dataset_name}")
            #    log.write(f" technique: {techniques_names[i]}")
            #    log.write(f" Environment: {venv_name}\n")
            #    log.write(f"Hour: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            #    log.write(f" Duration: {int(hours)}h {int(minutes)}m {int(seconds)}s\n")
            #    log.write(f"-"*20)
            #    log.write(f"\nResults: \n{results}\n")
            #    log.write(f"-"*20)
            #    log.write(f"\nerrors:\n")
            #    for ms in errors:
            #        log.write(f"{ms}\n")
            #    log.write(f"-"*20)
            #    log.write(f"\noutput:\n")
            #    for ms in output:
            #        log.write(f"{ms}\n")
            #    log.write("=" * 40 + "\n")

        else:
            print(f"{evaluation_name} for {dataset_name} done ")
        return 
    def run_evaluation_script(self, venv_name, path_evaluation, aligned_dataset_path, deidentified_dataset_path, pairs=[], save_path="./out.csv"):
        conda_sh_path = os.path.expanduser(self.__CONDA_DOT_SH_PATH)
        
        if not os.path.exists(conda_sh_path):
            print("conda.sh path does'nt exist, please change it in run_script() in deid_toolkit.py")
        
        
        command = (f"source {conda_sh_path} && "
                   f"conda activate {venv_name} && " 
                   f"python -u {path_evaluation} {aligned_dataset_path} {deidentified_dataset_path} ")
       
        command += f"--impostor_pairs_filepath {pairs[0]} " # add impostor file to the command
        command += f"--genuine_pairs_filepath {pairs[1]} " # add genuine file path to the command
        command += f"--save_path {save_path} " # add save path
        try:
            # execute the evaluation
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
        return 
    def _build_table_for_metrics(self,data):
        def _getHeaders(data):
            all_keys = set()  # Usar un set para evitar duplicados
            for evaluation_technique in data.values():
                for dataset in evaluation_technique.values():
                    for technique in dataset.values():
                        for metric in technique.values():
                            all_keys.update(metric.keys())  # Añadir todas las métricas directamente
            return sorted(list(all_keys))
        def _getRows(data, headers):
            rows = []
            for evaluation_technique in data.values():
                for dataset_name, dataset in evaluation_technique.items():
                    for technique_name,technique in dataset.items():
                        for metric_name, metric in technique.items():
                            row = [dataset_name, f"{technique_name}",f"{Fore.LIGHTYELLOW_EX}{metric_name}{Fore.RESET}"]
                            row += [f"{Fore.GREEN}{metric.get(key, '-')}{Fore.RESET}" for key in headers[3:]]
                            rows.append(row)
            return rows
        headers =["dataset", "technique", "metric",]
        headers += _getHeaders(data)
        rows = _getRows(data, headers)
        return rows, headers
    
