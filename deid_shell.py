import os  # os module provides a way of using operating system dependent functionality
import cmd
from turtle import color  # cmd is a module to create line-oriented command interpreters
from colorama import Fore  # color text
from tqdm import tqdm

import shutil

from modules.utils import ConfigManager

from modules import Datasets #1) 
from modules import Preprocessing #2)
from modules import Environments #3)
from modules import Techniques #4)
from modules import Evaluations # 5)
from modules import Visualization #6

class DeidShell(cmd.Cmd):
    intro = "Welcome to DeID-ToolKit.   Type help or ? to list commands.\n"
    prompt = "(deid) "
    file = None
    
    def __init__(self):
        super().__init__()
        self.root_dir = ConfigManager.get_instance().root_dir

        folder_datasets =ConfigManager.get_instance().FOLDER_DATASET
        folder_environments = ConfigManager.get_instance().FOLDER_ENVIRONMENTS
        folder_techniques = ConfigManager.get_instance().FOLDER_TECHNIQUES
        folder_evaluation = ConfigManager.get_instance().FOLDER_EVALUATION
        folder_visualization = ConfigManager.get_instance().FOLDER_VISUALIZATION
        #Initialize datasets
        self.datasets = Datasets(folder_datasets)
        self.datasets.initial_update(os.path.join(self.root_dir,folder_datasets,"aligned"),
                                os.path.join(self.root_dir,folder_datasets,"original"))
        #Initialize Preprocessing
        self.preprocessing = Preprocessing("Preprocessing")
        #Initialize Environments
        self.environments = Environments(folder_environments)
        self.environments.initial_update(os.path.join(self.root_dir, folder_environments))
        #Initialize techniques
        self.techniques =  Techniques(folder_techniques)
        self.techniques.initial_update(os.path.join(self.root_dir,folder_techniques))
        #Initialize Evaluations
        self.evaluation = Evaluations(folder_evaluation)
        self.evaluation.initial_update(os.path.join(self.root_dir,folder_evaluation))
        #Initialize Visualization
        self.visualization = Visualization(folder_visualization)
        self.visualization.initial_update(os.path.join(self.root_dir, folder_visualization))
        
        #self.logs_initial_update(
        #   os.path.join(self.root_dir,self.logs_dir, "preprocessing"),
        #    os.path.join(self.root_dir,self.logs_dir, FOLDER_TECHNIQUES),
        #    os.path.join(self.root_dir,self.logs_dir, FOLDER_EVALUATION),
        #    os.path.join(self.root_dir,self.logs_dir, FOLDER_VISUALIZATION)) #can log output, just add the path separate by ","
        #loads modules.yml
        #module_settings_file = config.get("settings", "modules_file") #get the modules settings filename from config.ini
        #self.modules_settings = self.get_modules_settings(module_settings_file)

    def do_exit(self, arg):
        "Exit the shell:  EXIT"
        self.close()
        return True  # self.do_bye(arg)

    def do_set_root(self, arg):
        "Set the root directory:  SET_ROOT /path/to/root"
        self.root_dir = arg
        print(Fore.CYAN + "Root directory set to: ", self.root_dir, Fore.RESET)
    #TODO: use properties to update the config file
    def do_root(self, arg):
        "Print the current root directory:  ROOT"
        if self.root_dir is None:
            print(Fore.RED + "Root directory not set, set it with SET_ROOT", Fore.RESET)
        else:
            print(Fore.CYAN + "Current root directory: ", self.root_dir, Fore.RESET)


    def do_selection(self,arg):
        selected_datasets = self.datasets.get_selection()
        selected_techniques =self.techniques.get_selection()
        selected_evaluation =self.evaluation.get_selection()
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
    def do_techniques(self,arg):
        "List all techniques in root:  TECHNIQUES"
        self.techniques.do_list()

    def do_evaluation(self,arg):
        "List all evaluation methods in root:  LIST_EVALUATION"
        self.evaluation.do_list()
        
    def do_datasets(self, arg):
        "List all datasets:  DATASETS"
        self.datasets.do_list()
    def do_environments(self, arg):
        "List all datasets:  ENVIRONMENTS"
        self.environments.do_list()

    def do_select(self, arg):
        "Select datasets, techniques, evaluation:  SELECT datasets|techniques|evaluation"

        if not arg:
            arg = "*"
            # print(Fore.RED + 'Please provide an argument. Type "help select" for more information', Fore.RESET)

        switcher = {
            "datasets": self.datasets.do_select,
            "techniques": self.techniques.do_select,
            "evaluation": self.evaluation.do_select,
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
            "preprocess": self.preprocessing.do_run,
            "environments": self.environments.do_run,
            "techniques": self.techniques.do_run,
            "evaluation": self.evaluation.do_run,
            "visualize": self.visualization.do_run,
        }

        switcher["*"] = lambda arg: [
            switcher[option](arg) for option in switcher.keys() if option != "*" # avoid the recursión
        ]  # run all

        # first split by " " and then by "|"
        current_args = arg.split(" ")[0]
        remaining_args = arg.split(" ")[1:]

        # split args by | and run each
        args = current_args.split("|")

        for arg in args:
            if arg in switcher:
                switcher[arg](remaining_args)

       # def run_preprocess_normalization(self, arg):
    #    'Run normalization:  RUN_PREPROCESS_NORMALIZATION'
    #    print('Running normalization')
#-----------------------------------------------------------------------------   
# change this
#-----------------------------------------------------------------------------   
    def logs_initial_update(self, *logs_folders):
        #if log path, exist, otherwise, create one
        for log_folder in logs_folders:
            if os.path.exists(log_folder):
                shutil.rmtree(log_folder)
            os.makedirs(log_folder, exist_ok=True)
            

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