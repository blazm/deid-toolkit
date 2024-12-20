from modules.utils.PipelineStage import IPipelineStage
from modules.utils.ErrorHandler import DeidtoolkitError
from modules.utils import ConfigManager  
import subprocess
import os

from colorama import Fore  # color text

class Environments(IPipelineStage):
    def __init__(self, stage_name):
        super().__init__(stage_name)
        self.__FOLDER_ENVIRONMENTS = ConfigManager.get_instance().FOLDER_ENVIRONMENTS
        self.package_manager_key = ConfigManager.get_instance().package_manager

    def initial_update(self, environments_folder):
        # Check if the config section for techniques exists; if not, create it
        if not self.config.has_section("Available Environments"):
            self.config.add_section("Available Environments")
            self.config.set("Available Environments", "")
        # Process the evaluation folder
        if os.path.exists(environments_folder):
            environments = os.listdir(environments_folder)
            environments.sort()
            for env in environments:
                # Check if it's a file and not a directory
                if (os.path.isfile(os.path.join(environments_folder,env)) 
                    and (env.endswith(".yml"))):
                        # Remove the extension
                        env_name = env.replace(".yml","")
                        self.config.set("Available Environments", env_name, env_name)# 2) The .yml file should be the same as well as the technique or evaluation name
        else:
            print(Fore.RED + 'Environments directory not found. Does the ROOT_DIR ({0}) have a folder named environments?'.format(self.root_dir), Fore.RESET)
        
        configini_filename = ConfigManager.get_instance().filename_config_toolkit
        # Save the configuration to the file
        with open(configini_filename, "w") as configfile:
            self.config.write(configfile)
    def do_list(self, *args):
        # Retrieve available environments  from the configuration
        config_environments = self.config.items("Available Environments")#
        environments:dict = {} # will return this value with a dicctionary format
        if not config_environments:
            print("No custom environments available")
            return
        
        # Check if the root directory is set
        if self.root_dir is None:
            print(Fore.RED + "Root directory not set, set it with SET_ROOT", Fore.RESET)
            return
        
        # Print instructions for environments  selection
        print(Fore.CYAN + "[Available environments]", Fore.RESET)

        # Display available environments
        for evaluation_name, env_name in config_environments:
            evaluation_name = evaluation_name.replace(".py","") #remove the extention.py 
            environments[evaluation_name]= env_name.replace(".yml","")
            print(Fore.LIGHTBLUE_EX + "\t" + evaluation_name + " : " + environments[evaluation_name], Fore.RESET)
        return environments
    def do_select(self,arg):
        raise DeidtoolkitError(f"Selection for environments have not been implemented yet")
    def get_selection(self):
        raise DeidtoolkitError("get_selection method have not been implemented for environments")        
    def do_run(self, *arg):
        """Create the environments before the techniques stage, this create environments for """
        selected_techniques_names = self.config.get("selection", "techniques").split()
        selected_evaluation_names =  self.config.get("selection", "evaluation").split()
        for technique_name in selected_techniques_names: 
            venv_exists = self.check_and_create_conda_env(technique_name)
        for evaluation_name in selected_evaluation_names: 
            venv_name = self.config.get("Available Environments",evaluation_name, fallback=evaluation_name)
            venv_exists = self.check_and_create_conda_env(venv_name)
        #TODO list all the conda env for evaluation and techniques and create them one by one
        pass
    def check_and_create_conda_env(self, env_name):
        env_names = Environments.get_system_environments()
        if env_name in env_names:
            print(f"'{env_name}' environment already exists")
            return True 
        else:
            print(f"'{env_name}' environment does not exist")
            yaml_file = os.path.join(self.root_dir,self.__FOLDER_ENVIRONMENTS ,env_name+".yml")
            if os.path.isfile(yaml_file):
                try:
                    if self.package_manager_key == 'mamba':
                        subprocess.check_call([self.package_manager_key, 'env', 'create', '-f', yaml_file, "--prefix", f"~/miniforge3/envs/{env_name}"])
                    else:
                        subprocess.check_call([self.package_manager_key, 'env', 'create', '-f', yaml_file, "--prefix", f"~/anaconda3/envs/{env_name}"])
                    print(f"'{env_name}' environment have been created")
                    return True
                except subprocess.CalledProcessError as e:
                    print(f"Error occurred creating '{env_name}' environment: {e}")
            else:
                print(f"'{yaml_file}' does not exist. '{env_name}' cannot be created.\nUsing the toolkit environment")
                return False
    @staticmethod 
    def isEnvironmentExist(venv:str)->list: 
        envs_list = Environments.get_system_environments()
        return venv in envs_list


    @staticmethod
    def get_system_environments() -> list:
        envs_list = subprocess.check_output([ConfigManager.get_instance().package_manager, 'env', 'list']).decode('utf-8').split('\n')
        env_name_from_list =''
        env_names =[]
        for line in envs_list:
            if line:
                env_name_from_list = line.split()[0]
                # print(f"env_name_from_list : {env_name_from_list}")
                env_names.append(env_name_from_list)
        return env_names