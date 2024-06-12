import os  # os module provides a way of using operating system dependent functionality
import cmd  # cmd is a module to create line-oriented command interpreters
from colorama import Fore  # color text
from tqdm import tqdm

FOLDER_DATASETS = "datasets/original"
FOLDER_TECHNIQUES = "techniques"
FOLDER_EVALUATION = "evaluation"
FOLDER_VISUALIZATION = "visualization"
FOLDER_DATASET_ALIGNED_SAVE = "datasets/aligned"
FOLDER_BLUR = "datasets/blurred"
FOLDER_PIXEL = "datasets/pixelized"


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
        datasets_initial_update(self,os.path.join(self.root_dir,FOLDER_DATASET_ALIGNED_SAVE),
                                os.path.join(self.root_dir,FOLDER_DATASETS))
        techniques_initial_update(self,os.path.join(self.root_dir,FOLDER_TECHNIQUES))

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
        get_available_datasets(self,arg)

    """           
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
    """

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

        switcher["*"] = lambda arg: [
            switcher[option](arg) for option in switcher.keys()
        ]  # run all

        args = arg.split("|")
        for arg in args:
            if arg in switcher:
                switcher[arg](arg)

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
        
        if not self.config.has_section("selection"):
            print("No datasets selected.")
            return

        selected_datasets = self.config.get("selection", "datasets").split()
        datasets_path = os.path.join(self.root_dir, FOLDER_DATASETS)

        if not os.path.exists(datasets_path):
            print(f"Datasets directory not found: {datasets_path}")
            return

        datasets = os.listdir(datasets_path)

        for index in selected_datasets:
            try:
                dataset_name = datasets[int(index)]
            except (ValueError, IndexError):
                print(f"Invalid dataset index: {index}")
                continue

            dataset_path = os.path.join(datasets_path, dataset_name, "img")
            dataset_save_path = os.path.join(self.root_dir,FOLDER_DATASET_ALIGNED_SAVE, dataset_name)

            if not os.path.exists(dataset_save_path):
                os.makedirs(dataset_save_path)

            if os.path.exists(os.path.join('root_dir/datasets/mirrored', dataset_name)):
                dataset_path = os.path.join('root_dir/datasets/mirrored', dataset_name)

            print(f"Aligning dataset: {dataset_name}")
            print(f"Source path: {dataset_path}")
            print(f"Save path: {dataset_save_path}")

            try:
                align_face_mtcnn.main(dataset_path=dataset_path, dataset_save_path=dataset_save_path,dataset_name=dataset_name)
                print(f"Successfully aligned dataset: {dataset_name}")
            except Exception as e:
                print(f"Error aligning dataset {dataset_name}: {e}")

    # def run_preprocess_normalization(self, arg):
    #    'Run normalization:  RUN_PREPROCESS_NORMALIZATION'
    #    print('Running normalization')

    def run_techniques(self, arg):
        "Run selected techniques on selected datasets: RUN_TECHNIQUES"
        print("Running techniques")

        if not self.config.has_section("selection"):
            print("No selection section in configuration.")
            return

        if not self.config.has_option("selection", "techniques") or not self.config.has_option("selection", "datasets"):
            print("No datasets or techniques selected.")
            return

        selected_datasets = self.config.get("selection", "datasets").split()
        selected_techniques = self.config.get("selection", "techniques").split()

        datasets_available = self._list_available_datasets()
        techniques_available = self._list_available_techniques()

        for technique_index in selected_techniques:
            try:
                technique_name = techniques_available[int(technique_index)]
                technique_module = self._import_technique(technique_name)
                for dataset_index in selected_datasets:
                    try:
                        dataset_name = datasets_available[int(dataset_index)]
                        self._process_dataset_with_technique(technique_name, technique_module, dataset_name)
                    except (ValueError, IndexError) as e:
                        print(f"Invalid dataset index: {dataset_index}. Error: {e}")
            except (ValueError, IndexError) as e:
                print(f"Invalid technique index: {technique_index}. Error: {e}")
                 

        
        # TODO: every technique must have a python script that can be run and deidentify either a single file or a directory
        # the script should be able to take input and output directories as arguments

    def run_evaluation(self, arg):
        "Run evaluation:  RUN_EVALUATION"
        print("Running evaluation")
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
            selected_datasets = self.config.get("selection", "datasets").split()
            datasets_path = os.path.join(self.root_dir, FOLDER_DATASETS)
            datasets_names = []
            if os.path.exists(datasets_path):
                datasets = os.listdir(datasets_path)
            for index in selected_datasets:
                try:
                    datasets_names.append(datasets[int(index)])
                except ValueError:
                    print("Invalid dataset index:", index)
                except IndexError:
                    print("Dataset index out of range:", index)
            print(datasets_names)
            generate_img_pairs_all.main(datasets_names)
        else:
            print("No datasets selected.")

    def select_datasets(self, arg):
        "List and interactively let user select datasets: SELECT_DATASETS"
        available_datasets = get_available_datasets(self,arg)
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

    def select_techniques(self, arg):
        "List and interactively let user select techniques: SELECT_TECHNIQUES"
        available_techniques = get_available_techniques(self,arg)

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

    def select_evaluation(self, arg):
        "List and interactively let user select evaluation modes:  SELECT_EVALUATION"
        if self.root_dir is None:
            print(Fore.RED + "Root directory not set, set it with SET_ROOT", Fore.RESET)
        else:
            print(Fore.CYAN + "[Evaluation]", Fore.RESET)
            evaluation_path = os.path.join(self.root_dir, FOLDER_EVALUATION)
            if os.path.exists(evaluation_path):
                evaluation = os.listdir(evaluation_path)
                for i, eval in enumerate(evaluation):
                    print(Fore.LIGHTYELLOW_EX + "\t" + str(i) + ". " + eval, Fore.RESET)
                print(
                    Fore.CYAN
                    + "Select evaluation by entering their numbers separated by space",
                    Fore.RESET,
                )
                selected_evaluation = input("Selection: ").split()
                for i in selected_evaluation:
                    try:
                        print(
                            Fore.LIGHTYELLOW_EX
                            + "\t"
                            + str(i)
                            + ". "
                            + evaluation[int(i)],
                            Fore.RESET,
                        )
                    except:
                        print(Fore.RED + "Invalid evaluation number: ", i, Fore.RESET)

                # create config section if it doesn't exist
                if not self.config.has_section("selection"):
                    self.config.add_section("selection")
                # save selected evaluation to config
                self.config.set(
                    "selection", "evaluation", " ".join(selected_evaluation)
                )
                # save config
                with open("config.ini", "w") as configfile:
                    self.config.write(configfile)
            else:
                print(
                    Fore.RED
                    + 'Evaluation directory not found. Does the ROOT_DIR ({0}) have folder named "evaluation"?'.format(
                        self.root_dir
                    ),
                    Fore.RESET,
                )

    def do_techniques(self, arg):
        "List all techniques in root:  TECHNIQUES"
        get_available_techniques(self,arg)

    def do_evaluation(self, arg):
        "List all evaluation methods:  LIST_EVALUATION"
        if self.root_dir is None:
            print(Fore.RED + "Root directory not set, set it with SET_ROOT", Fore.RESET)
        else:
            print(Fore.CYAN + "[Evaluation]", Fore.RESET)
            evaluation_path = os.path.join(self.root_dir, "evaluation")
            if os.path.exists(evaluation_path):
                evaluation_list = os.listdir(evaluation_path)
                for i, eval in enumerate(evaluation_list):
                    print(Fore.LIGHTYELLOW_EX + "\t" + str(i) + ". " + eval, Fore.RESET)
            else:
                print(
                    Fore.RED
                    + 'Evaluation directory not found. Does the ROOT_DIR ({0}) have folder named "evaluation"?'.format(
                        self.root_dir
                    ),
                    Fore.RESET,
                )

    def _list_available_datasets(self):
        path = os.path.join(self.root_dir,FOLDER_DATASET_ALIGNED_SAVE)
        if os.path.exists(path):
            return os.listdir(path)
        else:
            print(f"Dataset directory not found: {path}")
            return []

    def _list_available_techniques(self):
        techniques_path = os.path.join(self.root_dir,FOLDER_TECHNIQUES)
        if os.path.exists(techniques_path):
            return os.listdir(techniques_path)
        else:
            print(f"Techniques directory not found: {techniques_path}")
            return []

    def _import_technique(self, technique_name):
        try:
            module_name = technique_name.replace('.py', '')
            module = __import__(f"root_dir.techniques.{module_name}", fromlist=[''])
            print(f"{technique_name} imported")
            return module
        except ImportError as e:
            print(f"Error importing {technique_name}: {e}")
            return None

    def _process_dataset_with_technique(self, technique_name, technique_module, dataset_name):
        if technique_module is None:
            print(f"Technique module is None for {technique_name}")
            return

        dataset_path = os.path.join(self.root_dir,FOLDER_DATASET_ALIGNED_SAVE, dataset_name)
        if technique_name == 'pixel.py':
            dataset_save_path = os.path.join(self.root_dir,FOLDER_PIXEL, dataset_name)
        elif technique_name == 'blur.py':
            dataset_save_path = os.path.join(self.root_dir,FOLDER_BLUR, dataset_name)
        else:
            print(f"Unknown technique: {technique_name}")
            return

        if not os.path.exists(dataset_save_path):
            os.makedirs(dataset_save_path)

        print(f"Processing dataset: {dataset_name} | Source path: {dataset_path} | Save path: {dataset_save_path}")
        
        images = os.listdir(dataset_path)
        for img in tqdm(images, desc=f"Processing {dataset_name}"):
            input_path = os.path.join(dataset_path, img)
            output_path = os.path.join(dataset_save_path, img)
            try:
                technique_module.main(img_path=input_path, output_path=output_path)
                #print(f"Processed {img} with {technique_name}")
            except Exception as e:
                print(f"Error processing image {img} with {technique_name}: {e}")

             

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
            # Check if it's a file and not a directory (pycache i s problem)
            if os.path.isfile(os.path.join(techniques_folder,technique)):
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


def get_available_datasets(self,arg):
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


def get_available_techniques(self, arg):
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
