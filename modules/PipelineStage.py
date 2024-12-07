from abc import ABC,abstractmethod

class IPipelineStage(ABC):
    def __init__(self, stage_name):
        self.STAGE_NAME  = stage_name
    @abstractmethod
    def initial_update(self, folder):
        """implement this method is mandatory"""
        pass

    @abstractmethod
    def get_available(self):
        """implement this method is mandatory"""
        pass

    @abstractmethod
    def select(self, arg):
        """implement this method is mandatory"""
        pass
    
    @abstractmethod
    def run(self):
        """implement this method is mandatory"""
        pass
    