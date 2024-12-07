from abc import ABCMeta, abstractmethod
from configparser import ConfigParser
import os
#this is new import from branch "modules" to manage modules.yml
import yaml
from easydict import EasyDict as edict
from .ErrorHandler import DeidtoolkitError

CONDA_DOT_SH_PATH = "~/miniforge3/etc/profile.d/conda.sh"
FOLDER_DATASET = "datasets"
FOLDER_TECHNIQUES = "techniques"
FOLDER_EVALUATION = "evaluations"
FOLDER_VISUALIZATION = "visualization"
FOLDER_ENVIRONMENTS = "environments"
FOLDER_RESULTS = "results"

ROOT_DIR = "root_dir"
LOGS_DIR = "logs_dir"

class ConfigManager():
    
    __instance = None

    @staticmethod
    def get_instance():
        if ConfigManager.__instance == None: 
            ConfigManager("root_dir", "logs_dir") #TODO add values
        return ConfigManager.__instance
    def __init__(self, config_toolkit_filename):
        if ConfigManager.__instance != None: 
            raise Exception("ConfigManager cannot be instantiated more than once")
        else: 
            #asign values
            self.root_dir = ROOT_DIR
            self.logs_dir = LOGS_DIR
            self.FOLDER_DATASET = FOLDER_DATASET
            self.FOLDER_TECHNIQUES = FOLDER_TECHNIQUES
            self.FOLDER_EVALUATION = FOLDER_EVALUATION
            self.FOLDER_VISUALIZATION = FOLDER_VISUALIZATION
            self.FOLDER_ENVIRONMENTS = FOLDER_ENVIRONMENTS
            self.FOLDER_RESULTS = FOLDER_RESULTS
            self.CONDA_DOT_SH_PATH = CONDA_DOT_SH_PATH
            #1) load toolkit configuration
            self.filename_config_toolkit = config_toolkit_filename
            self.config_toolkit = self._read_config_toolkit(filename=config_toolkit_filename)
            #2) load pipeline .yml configuration
            config_module_filename= self.config_toolkit.get("settings", "modules_file")
            self.config_modules = self._read_module_settings(config_module_filename)
            ConfigManager.__instance = self
    def _read_config_toolkit(self,filename):
        config = ConfigParser()
        if os.path.exists(filename):
            config.read(filename)
            return config
        else:
            raise FileNotFoundError(f"Config file {filename} not found.")
    def _read_config_modules(self, filename):
        #load the yml_file
        required_keys = ["techniques", "evaluations", "visualization", "datasets"]  

        with open(filename, 'r') as file:
            data = yaml.safe_load(file) #avoid executing malicious code
            data_edict = edict(data) #loads with easy dict
       
        for key in required_keys:
            # Verify if the key is in required keys which is mandatory
            if key not in data or not data[key]:
                raise DeidtoolkitError(
                    f"Error loading {filename}: Missing values for '{key}' key.",
                    module="loading module settings file",
                    details=f"Please go to {filename} and add {key} with one or more entries."
                )

            # Verify if some of the methods is None
            for method, settings in data[key].items():
                if settings is None:
                    raise DeidtoolkitError(
                        f"Error loading {filename}: '{method}' in '{key}' is None.",
                        module="loading module settings file",
                        details=f"Please update or remove '{method}' from '{key}' in {filename}."
            )
        return data_edict #return the yml in a json format
 
    