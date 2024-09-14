import os  # os module provides a way of using operating system dependent functionality
import cmd  # cmd is a module to create line-oriented command interpreters
from colorama import Fore  # color text
from tqdm import tqdm
import subprocess
import select



FOLDER_DATASET = "datasets"
FOLDER_TECHNIQUES = "techniques"
FOLDER_EVALUATION = "evaluation"
FOLDER_VISUALIZATION = "visualization"


class DeidShell(cmd.Cmd):
    intro = "Welcome to DeID-ToolKit.   Type help or ? to list commands.\n"
    prompt = "(deid) "
    file = None
    root_dir = None
    config = None
   

    def __init__(
        self,
    ):  # , completekey: str = "tab", stdin: os.IO[str] | None = None, stdout: os.IO[str] | None = None) -> None:
        super().__init__()  # completekey, stdin, stdout)
        self.config = None


    def __init__(self, config):
        super().__init__()
        self.config = config
        self.root_dir = config.get("settings", "root_dir")
        self.datasets_initial_update(os.path.join(self.root_dir,FOLDER_DATASET,"aligned"),
                                os.path.join(self.root_dir,FOLDER_DATASET,"original"))
        self.techniques_initial_update(os.path.join(self.root_dir,FOLDER_TECHNIQUES))
        self.evaluation_initial_update(os.path.join(self.root_dir,FOLDER_EVALUATION))


    def do_exit(self, arg):
        "Exit the shell:  EXIT"
        self.close()
        return True  # self.do_bye(arg)


    def do_set_root(self, arg):
        "Set the root directory:  SET_ROOT /path/to/root"
        self.root_dir = arg
        print(Fore.CYAN + "Root directory set to: ", self.root_dir, Fore.RESET)


    def do_root(self, arg):
        "Print the current root directory:  ROOT"
        if self.root_dir is None:
            print(Fore.RED + "Root directory not set, set it with SET_ROOT", Fore.RESET)
        else:
            print(Fore.CYAN + "Current root directory: ", self.root_dir, Fore.RESET)


    def do_datasets(self, arg):
        "List all datasets:  DATASETS"
        self.get_available_datasets()

    def do_selection(self,arg):
        selected_datasets = self.config.get("selection","datasets").split(" ")
        selected_techniques =self.config.get("selection","techniques").split(" ")
        selected_evaluation =self.config.get("selection","evaluation").split(" ")
        print(Fore.LIGHTYELLOW_EX + "Current selection :", Fore.RESET )#\n Datasets : {selected_datasets} \n Techniques : {selected_techniques} \n Evaluation : {selected_evaluation}")
        print(Fore.LIGHTGREEN_EX + "Datasets:",Fore.RESET)
        for i in enumerate(selected_datasets):
            index = int(i[0])
            print(Fore.LIGHTGREEN_EX + "\t"+str(index)+". "+selected_datasets[index],Fore.RESET)
        print(Fore.LIGHTCYAN_EX + "Techniques:",Fore.RESET)
        for i in enumerate(selected_techniques):
            index = int(i[0])
            print(Fore.LIGHTCYAN_EX + "\t"+str(index)+". "+selected_techniques[index],Fore.RESET)
        print(Fore.LIGHTMAGENTA_EX + "Evaluation:",Fore.RESET)
        for i in enumerate(selected_evaluation):
            index = int(i[0])
            print(Fore.LIGHTMAGENTA_EX + "\t"+str(index)+". "+selected_evaluation[index],Fore.RESET)

    """""          
    def do_set(self, arg):
        'Set configuration:  SET key value'
        args = arg.split()
        
        if len(args) == 2:
            if not self.config.has_section('user_values'):
                self.config.add_section('user_values')
            
            self.config.set('user_values', args[0], args[1])
            with open('config.ini', 'w') as configfile:
                self.config.write(configfile)
        else:
            print(Fore.RED + 'Invalid number of arguments. Type "help set" for more information', Fore.RESET)

                    
    def do_get(self, arg):
        'Get configuration:  GET key'
        args = arg.split()
        if len(args) == 1:
            print(Fore.CYAN + self.config.get('user_values', args[0]), Fore.RESET)
        else:
            print(Fore.RED + 'Invalid number of arguments. Type "help get" for more information', Fore.RESET)
    """""


    def do_select(self, arg):
        "Select datasets, techniques, evaluation:  SELECT datasets|techniques|evaluation"

        if not arg:
            arg = "*"
            # print(Fore.RED + 'Please provide an argument. Type "help select" for more information', Fore.RESET)

        switcher = {
            "datasets": self.select_datasets,
            "techniques": self.select_techniques,
            "evaluation": self.select_evaluation,
        }

        def run_all():
            for option in switcher.values():
                option()

        if arg == "*":
            run_all()
        else:
            args = arg.split("|")
            for arg in args:
                if arg in switcher:
                    switcher[arg]()
                else:
                    print(f"Unknown argument: {arg}. Please provide a valid argument (datasets, techniques, evaluation).")



    def do_run(self, arg):
        "Run selected preprocessing, techniques, evaluation, visualize:  RUN"

        # check if empty arg and print help
        if not arg:
            arg = "*"
            # print(Fore.RED + 'Please provide an argument. Type "help run" for more information', Fore.RESET)
            # return

        switcher = {
            "preprocess": self.run_preprocess,
            "techniques": self.run_techniques,
            "evaluation": self.run_evaluation,
            "visualize": self.run_visualize,
            "generate_pairs": self.run_generate_pairs,
        }

        switcher["*"] = lambda arg: [
            switcher[option](arg) for option in switcher.keys()
        ]  # run all

        # first split by " " and then by "|"
        current_args = arg.split(" ")[0]
        remaining_args = arg.split(" ")[1:]

        # split args by | and run each
        args = current_args.split("|")

        for arg in args:
            if arg in switcher:
                switcher[arg](remaining_args)


    def run_preprocess(self, arg):
        "Run preprocessing:  RUN_PREPROCESS"
        print("Running preprocessing")
        if not arg:
            arg = "*"
        preprocess_order = ["alignment"]

        switcher = {
            "alignment": self.run_preprocess_alignment,
            #'normalization': self.run_preprocess_normalization,
        }
        switcher["*"] = lambda arg: [
            switcher[option](arg) for option in switcher.keys()
        ]  # run all

        # # TODO: every preprocessing step must have a python script that can be run and preprocess either a single file or a directory
        # # the script should be able to take input and output directories as arguments

        for step in preprocess_order:
            switcher[step](arg)


    def run_preprocess_alignment(self, arg):
        "Run alignment:  RUN_PREPROCESS_ALIGNMENT"
        print("Running alignment")
        import align_face_mtcnn
        aligned_datasets = self.config.get("Available Datasets","aligned").split()
        selected_datasets_names = self.config.get("selection", "datasets").split()
        datasets_path = os.path.join(self.root_dir, FOLDER_DATASET,"original")

        dataset_names = ''

        if not os.path.exists(datasets_path):
            print(f"Datasets directory not found: {datasets_path}")
            return

        for dataset_name in selected_datasets_names:

            dataset_path = os.path.join(datasets_path, dataset_name, "img")
            dataset_save_path = os.path.join(self.root_dir,FOLDER_DATASET,"aligned", dataset_name)

            if not os.path.exists(dataset_save_path):
                os.makedirs(dataset_save_path)

            if os.path.exists(os.path.join(self.root_dir,FOLDER_DATASET,"mirrored", dataset_name)):
                dataset_path = os.path.join(self.root_dir,FOLDER_DATASET,"mirrored", dataset_name)

            print(f"Aligning dataset: {dataset_name}")
            print(f"Source path: {dataset_path}")
            print(f"Save path: {dataset_save_path}")

            try:
                align_face_mtcnn.main(dataset_path=dataset_path, dataset_save_path=dataset_save_path,dataset_name=dataset_name)
                print(f"Successfully aligned dataset: {dataset_name}")
                if dataset_name not in aligned_datasets:
                    aligned_datasets.append(dataset_name)
                    aligned_datasets.sort()
                    for i in aligned_datasets:
                        dataset_names+= (i+' ')
                    dataset_names = dataset_names.strip()
                    self.config.set("Available Datasets","aligned",dataset_names)
                    with open("config.ini", "w") as configfile:
                        self.config.write(configfile)   
            except Exception as e:
                print(f"Error aligning dataset {dataset_name}: {e}")

    # def run_preprocess_normalization(self, arg):
    #    'Run normalization:  RUN_PREPROCESS_NORMALIZATION'
    #    print('Running normalization')


    def run_techniques(self, arg):
        "Run selected techniques on selected datasets: RUN_TECHNIQUES"
        print("Running techniques")
        if not self.config.has_option("selection", "techniques") or not self.config.has_option("selection", "datasets"):
            print("No datasets or techniques selected.")
            return

        selected_datasets_names = self.config.get("selection", "datasets").split()
        selected_techniques_names = self.config.get("selection", "techniques").split()

        for technique_name in selected_techniques_names:
            try:
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


    def run_evaluation(self, arg):
        "Run evaluation:  RUN_EVALUATION"
        print("Running evaluation")
        #Check if datasets or evaluation methods are selected
        if not self.config.has_option("selection", "evaluation") or not self.config.has_option("selection", "datasets"):
            print("No datasets or evaluation selected.")
            return
        selected_datasets_names = self.config.get("selection", "datasets").split()
        selected_evaluation_names = self.config.get("selection", "evaluation").split()

        for evaluation_name in selected_evaluation_names:
            print(f"Evaluation method {evaluation_name}")
            try:
                #TODO check if need something before run evaluation for each dataset
                for dataset_name in selected_datasets_names:
                    try:
                        #TODO run evalatuation method for each technique like _process_dataset_with_evalatuation(technique, venv_name, dataset_name)
                        print(f"LOG: Evaluating dataset {dataset_name} with {evaluation_name}")
                    except (ValueError, IndexError) as e: 
                        print(f"Invalid dataset index: {dataset_name}. Error: {e}")
            except (ValueError, IndexError) as e:
                print(f"Invalid eevaluation method index: {evaluation_name}. Error: {e}")
        
        # TODO: every evaluation step must have a python script that can be run and preprocess either a single file or a directory
        # the script should be able to take input and output directories as arguments


    def run_visualize(self, arg):
        "Run visualization:  RUN_VISUALIZE"
        print("Running visualization")
        # TODO: every visualization step must have a python script that can be run and preprocess either a single file or a directory
        # the script should be able to take input and output directories as arguments


    def run_generate_pairs(self, arg):
        print("Generation of pairs on selected datasets")
        import generate_img_pairs_all

        if self.config.has_section("selection"):
            selected_datasets_names = self.config.get("selection", "datasets").split()
            FOLDER_LABELS = os.path.join(self.root_dir,FOLDER_DATASET,"labels")
            PAIRS_FOLDER = os.path.join(self.root_dir,FOLDER_DATASET,"pairs")
            generate_img_pairs_all.main(selected_datasets_names, FOLDER_LABELS,PAIRS_FOLDER)
        else:
            print("No datasets selected.")


    def select_datasets(self):
        "List and interactively let user select datasets: SELECT_DATASETS"
        available_datasets = self.get_available_datasets()
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

        # Save the configuration
        with open("config.ini", "w") as configfile:
            self.config.write(configfile)


    def select_techniques(self):
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


    def select_evaluation(self):
        "List and interactively let user select evaluation modes:  SELECT_EVALUATION"
        available_evaluations = self.get_available_evaluations()

        if not available_evaluations:
            print("No evaluation method available")
            return
        print(Fore.CYAN + "Select evaluation by entering their numbers separated by space", Fore.RESET)
        # Prompt the user to select techniques
        selected_evaluations_indices = input("Selection: ").split()

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

        # Save the configuration
        with open("config.ini", "w") as configfile:
            self.config.write(configfile)


    def do_techniques(self,arg):
        "List all techniques in root:  TECHNIQUES"
        self.get_available_techniques()


    def do_evaluation(self,arg):
        "List all evaluation methods in root:  LIST_EVALUATION"
        self.get_available_evaluations()


    def _list_available_datasets(self):
        path = os.path.join(self.root_dir,FOLDER_DATASET,"aligned")
        if os.path.exists(path):
            return os.listdir(path)
        else:
            print(f"Dataset directory not found: {path}")
            return []


    def _list_available_techniques(self):
        techniques_path = os.path.join(self.root_dir, FOLDER_TECHNIQUES)
        if os.path.exists(techniques_path):
            # Lister uniquement les fichiers avec l'extension .py
            return [f for f in os.listdir(techniques_path) if f.endswith('.py')]
        else:
            print(f"Techniques directory not found: {techniques_path}")
            return []


    def check_and_create_conda_env(self, env_name):
        envs_list = subprocess.check_output(['mamba', 'env', 'list']).decode('utf-8').split('\n')
        # print(f"env list : {envs_list}")
        # print(f"env_name: {env_name}")
        env_name_from_list =''
        env_names =[]
        for line in envs_list:
            if line:
                env_name_from_list = line.split()[0]
                # print(f"env_name_from_list : {env_name_from_list}")
                env_names.append(env_name_from_list)


        if env_name in env_names:
            print(f"'{env_name}' environment already exists")
            return True 
        else:
            print(f"'{env_name}' environment does not exist")
            yaml_file = os.path.join(self.root_dir,"techniques","environments",env_name+".yml")
            if os.path.isfile(yaml_file):
                try:
                    subprocess.check_call(['mamba', 'env', 'create', '-f', yaml_file, "--prefix", "/opt/conda/envs/"])
                    print(f"'{env_name}' environment have been created")
                    return True
                except subprocess.CalledProcessError as e:
                    print(f"Error occurred creating '{env_name}' environment: {e}")
            else:
                print(f"'{yaml_file}' does not exist. '{env_name}' cannot be created.\nUsing the toolkit environment")
                return False

    def _process_dataset_with_technique(self, technique_name, venv_name, dataset_name):
        answer = ''
        original_dataset_path = os.path.join(self.root_dir, FOLDER_DATASET,"original", dataset_name, "img")
        if technique_name is None:
            print(f"Technique module is None for {technique_name}")
            return

        aligned_dataset_path = os.path.join(self.root_dir, FOLDER_DATASET,"aligned", dataset_name)
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

        self.run_script(venv_name, technique_name, aligned_dataset_path, dataset_save_path)

    def run_script(self, venv_name, technique_name, aligned_dataset_path, dataset_save_path):
        conda_sh_path = os.path.expanduser("/opt/conda/etc/profile.d/conda.sh")

        if not os.path.exists(conda_sh_path):
            print("conda.sh path does'nt exist, please change it in run_script() in deid_toolkit.py")

        aligned_dataset_path = os.path.abspath(aligned_dataset_path)
        dataset_save_path = os.path.abspath(dataset_save_path)
        path_technique_folder = os.path.join(self.root_dir,FOLDER_TECHNIQUES)
        
        command = (
            f"source {conda_sh_path} && "
            f"conda activate {venv_name} && "
            f"cd {path_technique_folder} && "
            f"python -u {technique_name}.py {aligned_dataset_path} {dataset_save_path} "
        )
        
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
        

    def datasets_initial_update(self, aligned_folder, original_folder):
        datasets_name = ""

        # Check if the config section for datasets exists; if not, create it
        if not self.config.has_section("Available Datasets"):
            self.config.add_section("Available Datasets")
            self.config.set("Available Datasets", "Original", "")
            self.config.set("Available Datasets", "Aligned", "")

        # Process the original datasets
        if os.path.exists(original_folder):
            datasets = os.listdir(original_folder)
            datasets.sort()
            for dataset in datasets:
                if os.path.isdir(os.path.join(original_folder, dataset)):
                    datasets_name += (dataset + " ")
            # Remove the trailing space
            datasets_name = datasets_name.strip()
            self.config.set("Available Datasets", "Original", datasets_name)
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
            self.config.set("Available Datasets", "Aligned", datasets_name)
            datasets_name = ''
        else:
            print(Fore.RED + 'Aligned datasets directory not found. Does the ROOT_DIR ({0}) have a folder named "datasets"?'.format(self.root_dir), Fore.RESET)

        # Save the configuration to the file
        with open("config.ini", "w") as configfile:
            self.config.write(configfile)        


    def get_available_datasets(self):
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
        print(Fore.CYAN + "[Datasets]", Fore.RESET)

        # Display available datasets with appropriate colors
        for i, dataset in enumerate(available_datasets):
            if dataset in aligned_datasets:
                print(Fore.LIGHTGREEN_EX + "\t" + str(i) + ". " + dataset, Fore.RESET)
            else:
                print(Fore.LIGHTYELLOW_EX + "\t" + str(i) + ". " + dataset, Fore.RESET)

        
        return available_datasets


    def techniques_initial_update(self, techniques_folder):
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
        with open("config.ini", "w") as configfile:
            self.config.write(configfile)


    def get_available_techniques(self):
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

        # Display available techniques
        for i, technique in enumerate(techniques):
            print(Fore.LIGHTYELLOW_EX + "\t" + str(i) + ". " + technique, Fore.RESET)

        return techniques


    def evaluation_initial_update(self, evaluation_folder):
        evaluation_names = ""

        # Check if the config section for techniques exists; if not, create it
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
            print(Fore.RED + 'Evalauation directory not found. Does the ROOT_DIR ({0}) have a folder named "evaluation"?'.format(self.root_dir), Fore.RESET)

        # Save the configuration to the file
        with open("config.ini", "w") as configfile:
            self.config.write(configfile)


    def get_available_evaluations(self):
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

        # Display available techniques
        for i, evaluation in enumerate(evaluations):
            print(Fore.LIGHTYELLOW_EX + "\t" + str(i) + ". " + evaluation, Fore.RESET)

        return evaluations
    

    # arbitrary method to parse all other commands
    def default(self, line):
        "Called on an input line when the command prefix is not recognized"
        switcher = {
            "config": lambda: print(
                {
                    section: dict(self.config[section])
                    for section in self.config.sections()
                }
            ),
            #'select': self.do_select,
            #'run': self.do_run,
            #'exit': self.do_exit
            "server": lambda: print("Server is running on port 8080"),
        }

        if line in switcher:
            switcher[line]()

    # ----- record and playback -----
    def do_record(self, arg):
        "Save future commands to filename:  RECORD rose.cmd"
        self.file = open(arg, "w")

    def do_playback(self, arg):
        "Playback commands from a file:  PLAYBACK rose.cmd"
        self.close()
        with open(arg) as f:
            self.cmdqueue.extend(f.read().splitlines())

    def precmd(self, line):
        line = line.lower()
        if self.file and "playback" not in line:
            print(line, file=self.file)
        return line

    def close(self):
        if self.file:
            self.file.close()
            self.file = None

    def parse(arg):
        "Convert a series of zero or more numbers to an argument tuple"
        return tuple(map(int, arg.split()))