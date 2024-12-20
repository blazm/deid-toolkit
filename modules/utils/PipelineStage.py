from abc import ABC,abstractmethod
from . import ConfigManager
from .ErrorHandler import DeidtoolkitError
class IPipelineStage(ABC):
    def __init__(self, stage_name):
        self.STAGE_NAME  = stage_name
        self.config = ConfigManager.get_instance().config_toolkit
        self.module_settings = ConfigManager.get_instance().config_modules
        self.root_dir = ConfigManager.get_instance().root_dir
        self.filename_config_toolkit = ConfigManager.get_instance().filename_config_toolkit
    
    @abstractmethod
    def initial_update(self, *folder):
        """implement this method is mandatory"""
        pass
    @abstractmethod
    def do_list(self, *arg):
        """implement this method is mandatory to get the available methods"""
        pass

    @abstractmethod
    def do_select(self, *arg):
        """implement this method is mandatory"""
        pass
    @abstractmethod 
    def get_selection(self, *arg): 
        """implement this method is mandatory"""
        pass 
    
    @abstractmethod
    def do_run(self, *arg):
        """implement this method is mandatory"""
        pass
    