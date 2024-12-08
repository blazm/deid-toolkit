from modules.utils.ErrorHandler import DeidtoolkitError
from modules.utils.PipelineStage import IPipelineStage
from modules.utils import ConfigManager  
import subprocess
import os
import select
from colorama import Fore  # color text


class Visualization(IPipelineStage):
    def __init__(self, stage_name):
        super().__init__(stage_name)
        self.__FOLDER_VISUALIZATION = ConfigManager.get_instance().FOLDER_VISUALIZATION
    def initial_update(self, visualization_folder):
        if not self.config.has_section("Available Visualizations"):
            self.config.add_section("Available Visualizations")
        if os.path.exists(visualization_folder):
            visualization = os.listdir(visualization_folder)
            visualization.sort()
            for visual in visualization:
                if (os.path.isfile(os.path.join(visualization_folder, visual))
                    and (visual.endswith(".py"))):
                        visualization_name = visual.replace(".py","")
                        if not self.config.has_option("Available Visualizations", visualization_name):
                            self.config.set("Available Visualizations", visualization_name, "")
        else:
            print(Fore.RED + 'Visualization directory not found. Does the ROOT_DIR ({0}) have a folder named visualization?'.format(self.root_dir), Fore.RESET)
        configini_filename = ConfigManager.get_instance().filename_config_toolkit
        with open(configini_filename, "w") as configfile:
            self.config.write(configfile)
    def do_list(self, *arg):
        """
        Will return an dictionary with key (visualization python script name) and values in array corresponding to each evaluation name
        """
        config_visualizations = self.config.items("Available Visualizations")#
        visualization_dict:dict = {} # will return this value with a dicctionary format
        if not config_visualizations:
            print("No visualizations available")
            return

        if self.root_dir is None:
            print(Fore.RED + "Root directory not set, set it with SET_ROOT", Fore.RESET)
            return
        
        # Print instructions for environments  selection
        print(Fore.CYAN + "[Available  Visualizations]", Fore.RESET)

        # Display available environments
        for visualization_name, evaluation_names in config_visualizations:
            visualization_name = visualization_name.replace(".py","") #remove the extention.py 
            visualization_dict[visualization_name]= evaluation_names.split(" ")
            print(f"{Fore.LIGHTBLUE_EX}\t{visualization_name}:{visualization_dict[visualization_name]}", Fore.RESET)
        return visualization_dict
    def do_select(self, arg):
        raise DeidtoolkitError("do_select() method for visulizations have not been implemented")
    def get_selection(self):
        raise DeidtoolkitError("get_select() method for visulizations have not been implemented")
    def do_run(self):
        "Run visualization:  RUN_VISUALIZE"
        print("Running visualization")
        if (not self.config.has_option("selection", "evaluation") 
            or not self.config.has_option("selection", "datasets")
            or not self.config.has_option("selection","techniques")):
            print("No datasets or evaluation or techniques selected. Please, check your selected options")
            return
        #get all the selected items
        selected_evaluation_names = self.config.get("selection", "evaluation").split()
        selected_datasets_names = self.config.get("selection", "datasets").split()
        selected_techniques_names = self.config.get("selection", "techniques").split()

        if not selected_evaluation_names:
            print("No evaluation methods are selected")
            return
        print("select evaluation names:", selected_evaluation_names)
        available_visualization  = self.get_available_visualizations()
        for visualization, evaluations_list in available_visualization.items():
            #filter only the selected evaluatios in the configurarion list
            evaluations_list = [e for e in evaluations_list if e != '']
            if len(evaluations_list) == 0:
                print(f"No evaluation list in config.ini: {visualization}=")
                continue
            selected_evaluations = [evaluation for evaluation in evaluations_list if evaluation in selected_evaluation_names]
            path_visualization_script =  os.path.abspath(os.path.join(self.root_dir, self.__FOLDER_VISUALIZATION, visualization))
            path_to_save =  os.path.abspath(os.path.join(self.root_dir, "visuals"))
            if len(selected_evaluations) > 0: 
                #we need at least one evaluation, otherwise, will be skipped
                self.run_visualization_script(path_visualization_script, selected_evaluations,selected_datasets_names, selected_techniques_names ,path_to_save )
        
        
        # TODO: every visualization step must have a python script that can be run and preprocess either a single file or a directory
        # the script should be able to take input and output directories as arguments

    def run_visualization_script(self, path_visualization_script,evaluations:list, datasets:list, techniques:list, path_to_save  ):
        evaluations_args = ",".join(evaluations)
        datasets_args = ",".join(datasets)
        techniques_args = ",".join(techniques)
        command = (f"python -u {path_visualization_script}.py ")
        command += f"--evaluations {evaluations_args} "
        command += f"--datasets {datasets_args} "
        command += f"--techniques {techniques_args} "
        command += f"--path_save {path_to_save} "
        print(f"{Fore.LIGHTCYAN_EX}{'-' * 40}{Fore.RESET}")
        print(f"{Fore.LIGHTYELLOW_EX}{os.path.basename(path_visualization_script)}{Fore.RESET}")
        print(f"{Fore.LIGHTCYAN_EX}{'-' * 40}{Fore.RESET}")
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
        