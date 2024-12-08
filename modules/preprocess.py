from modules.utils.ConfigManager import ConfigManager
from modules.utils.PipelineStage import IPipelineStage
from modules.utils.ErrorHandler import DeidtoolkitError
import modules.utils.generate_img_pairs_all as generate_img_pairs_all
import modules.utils.align_face_mtcnn as align_face_mtcnn

import subprocess
import os
from colorama import Fore, Back  # color text

class Preprocessing(IPipelineStage):
    def __init__(self, stage_name):
        super().__init__(stage_name)
        self.__FOLDER_DATASET = ConfigManager.get_instance().FOLDER_DATASET
    def initial_update(self, *folder):
        raise DeidtoolkitError("initial_update() for Preprocessing have not been implemented")
    def do_select(self, *arg):
        raise DeidtoolkitError("do_select have not been implemented yet for Preprocessing")
    def do_list(self, *arg):
        raise DeidtoolkitError("do_list have not been implemented yet for Preprocessing")
    def get_selection(self, *arg):
        raise DeidtoolkitError("get_selection() have not been implemented yet for Preprocessing")
    def do_run(self , *arg):
        "Run preprocessing:  RUN_PREPROCESS"
        print(Back.GREEN, Fore.WHITE,"Running preprocessing",Back.RESET, Fore.RESET)
        if not arg:
            arg = "*"
        preprocess_order = ["alignment", "generate_image_pairs"]

        switcher = {
            "alignment": self.run_preprocess_alignment,
            "generate_image_pairs": self.run_generate_pairs
            #'normalization': self.run_preprocess_normalization,
        }
        switcher["*"] = lambda arg: [
            switcher[option](arg) for option in switcher.keys()
        ]  # run all

        # # TODO: every preprocessing step must have a python script that can be run and preprocess either a single file or a directory
        # # the script should be able to take input and output directories as arguments

        for step in preprocess_order:
            switcher[step](arg)
        return
    def run_preprocess_alignment(self, *arg):
        "Run alignment:  RUN_PREPROCESS_ALIGNMENT"
        print(Fore.GREEN,"--> Running alignment", Fore.RESET)
        aligned_datasets = self.config.get("Available Datasets","aligned").split()
        selected_datasets_names = self.config.get("selection", "datasets").split()
        datasets_path = os.path.join(self.root_dir,
                                    self.__FOLDER_DATASET,
                                    "original")
        dataset_names = ''

        if not os.path.exists(datasets_path):
            print(f"Datasets directory not found: {datasets_path}")
            return

        for dataset_name in selected_datasets_names:
            dataset_path = os.path.join(datasets_path, dataset_name, "img")
            dataset_save_path = os.path.join(self.root_dir,self.__FOLDER_DATASET,"aligned", dataset_name)

            if not os.path.exists(dataset_save_path):
                os.makedirs(dataset_save_path)

            if os.path.exists(os.path.join(self.root_dir,self.__FOLDER_DATASET,"mirrored", dataset_name)):
                dataset_path = os.path.join(self.root_dir,self.__FOLDER_DATASET,"mirrored", dataset_name)
            
            print(Fore.LIGHTWHITE_EX,f"Aligning dataset:", Fore.LIGHTBLACK_EX, dataset_name, Fore.RESET)
            print(Fore.LIGHTWHITE_EX,f"Source path: ",Fore.LIGHTBLACK_EX, dataset_path)
            print(Fore.LIGHTWHITE_EX,f"Save path: ", Fore.LIGHTBLACK_EX,dataset_save_path,  Fore.RESET)

            try:
                align_face_mtcnn.main(dataset_path=dataset_path, dataset_save_path=dataset_save_path,dataset_name=dataset_name)
                print(Fore.GREEN, f"Successfully aligned dataset: {dataset_name}",  Fore.RESET)
                if dataset_name not in aligned_datasets:
                    aligned_datasets.append(dataset_name)
                    aligned_datasets.sort()
                    for i in aligned_datasets:
                        dataset_names+= (i+' ')
                    dataset_names = dataset_names.strip()
                    self.config.set("Available Datasets","aligned",dataset_names)
                    with open(ConfigManager.get_instance().filename_config_toolkit, "w") as configfile:
                        self.config.write(configfile)   
            except Exception as e:
                print(f"Error aligning dataset {dataset_name}: {e}")
    
    def run_generate_pairs(self, *arg):
        print(Fore.GREEN,"--> Generation of pairs on selected datasets", Fore.RESET)
        if self.config.has_section("selection"):
            selected_datasets_names = self.config.get("selection", "datasets").split()
            FOLDER_LABELS = os.path.join(self.root_dir,self.__FOLDER_DATASET,"labels")
            PAIRS_FOLDER = os.path.join(self.root_dir,self.__FOLDER_DATASET,"pairs")
            generate_img_pairs_all.main(selected_datasets_names, FOLDER_LABELS,PAIRS_FOLDER)
        else:
            print(Fore.YELLOW,"No datasets selected.", Fore.RESET)

