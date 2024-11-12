from PIL import Image
import numpy as np
import os
import argparse
import shutil 

import pandas as pd



"""
This file will provide usefull functions for the scripts that performs evaluation techniques.
"""
OUTPUT_FOLDER = "./root_dir/evaluation/output"
ROOT_FOLDER = "./root_dir/evaluation/"
TEMP_DIR = "./root_dir/evaluation/tmp"


os.makedirs(os.path.abspath(TEMP_DIR), exist_ok=True)


#/evaluation/output"
def read_pairs_file(filepath:str):
    """
    This function recieve a generated genuine and impostor image pairs to extract it's content
    in a variable on python
    """
    imgs_a, ids_a, imgs_b, ids_b = [], [], [], []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            row = line.strip().split(" ")
            id_a, image_a, id_b, image_b = row
            imgs_a.append(image_a)
            ids_a.append(id_a)
            imgs_b.append(image_b)
            ids_b.append(id_b)
    return imgs_a, ids_a, imgs_b, ids_b
def convert_png_to_jpg(png_file_path, quality=95):
    with Image.open(png_file_path) as img:
        # Convert to RGB (to avoid issues with RGBA)
        rgb_img = img.convert('RGB')
        # Save the image in JPG format with specified quality
        jpgfile =png_file_path.replace("png", "jpg")
        rgb_img.save(jpgfile, quality=quality)  # High quality for minimal loss
    os.remove(png_file_path)
def read_args(): 
    """
    Function to parse command-line arguments for evaluating metrics.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Args to evaluate metrics")
    
    # Positional arguments
    parser.add_argument('aligned_path', type=str, help='Paths of the aligned datasets')
    parser.add_argument('deidentified_path', type=str, help='Paths of the deidentified datasets')

    # Optional arguments
    parser.add_argument('--save_path', type=str, help="Path file to save the results. If not exist, it will be created.")
    parser.add_argument('--impostor_pairs_filepath', type=str, help="Path to impostor files")
    parser.add_argument('--genuine_pairs_filepath', type=str, help="Path to genuine files")
    parser.add_argument('--dir_to_log', type=str, help="Path to directory to log", default="./root_dir/logs/evaluation/")
    
    
    # Parse arguments
    args = parser.parse_args()
    assert os.path.exists(args.aligned_path)
    assert os.path.exists(args.deidentified_path)
    if args.save_path is not None:
        assert args.save_path.endswith(".csv")
    if args.impostor_pairs_filepath is not None: 
        assert os.path.exists(args.impostor_pairs_filepath)
    if args.genuine_pairs_filepath is not None:
        assert os.path.exists(args.genuine_pairs_filepath)
    return args

    

def _resize(img0:Image, img1: Image):
    """
    This function recieve two pillow images as and arguments
    and resize the first one to the same as the seconda one.
    Arguments: 
        img0: to resize
        img1: the reference image.
    Returns:
        The img0 resized. 
    """
    # Resize img0 to match img1's size
    reference_size = img1.size 
    return img0.resize(reference_size)

def resize_if_different(img0:Image, img1:Image)->Image:
    """
    Checks if the two provided images are the same shape,
    if not, resize the first one; otherwise, return the same image.
    Arguments:
        img0: Pillow image to compare and resize.
        img1: Pillow image reference image to compare with the first one.
    Returns:
        The img0 resized or not.
    """
    if img0.size != img1.size:
        img0 = _resize(img0, img1)
    return img0

def get_output_filename(metric_name:str,aligned_path:str, deidentified_path:str)->str:
    """
    Given three argumnents, build a name for the output scores.
    Arguments:
        metric_name: the name of the evaluation technique
        aligned_path: is the location of aligned dataset path
        deidentified_path: the location of the deidentified dataset path, normally in the technique folder from datasets
    Returns:
        a str with the destination filename.
    """
    dataset_name = aligned_path.split("/")[-1]
    technique_name = deidentified_path.split("/")[-2]
    relative_path = f".{OUTPUT_FOLDER}/{metric_name}_{dataset_name}_{technique_name}.txt"
    absolute_output_path = os.path.abspath(relative_path)
    return absolute_output_path
def get_dataset_name_from_path(aligned_path:str):
    aligned_path = aligned_path.split("/")
    aligned_path.remove('')
    return aligned_path[-1]
def get_technique_name_from_path(deidentified_path:str):
    """
    Args:
        deidentified_path (str): this is mandatory because the deidentified path must contain
        the {technique}/{dataset} in the path, index: -2

    Returns:
        str: name of the technique
    """
    deidentified_path = deidentified_path.split("/")
    deidentified_path.remove('')
    return deidentified_path[-2]
def log_image_in_output(src_img, destination):
    if not os.path.exists(destination):
        os.makedirs(destination)
    shutil.move(src_img, destination)

def compute_mean_std(output_scores_file:str)-> tuple:
    """
    Takes the output scores filename (txt), open it and extract all the information to compute the mean and std using numpy arr
    Arguments:
        output_scores_filename: is the path to read the txt
    Returns: 
        a tuple of the mean as a first param, and std as a second param.
    """
    arr = np.loadtxt(output_scores_file)
    return arr.mean() , arr.std()

def log(log_filename, text):
    with open(log_filename, 'a') as log_file:
        log_file.write(text + '\n') 

class Metrics():
    metric_df = None
    def __init__(self, name_score:str):
        # Call the parent constructor (for DataFrame functionality)
        self.columns = ['img', f'{name_score}']
        # Store the required parameters as class attributes
        self.name_score = name_score
        self.metric_df = pd.DataFrame(columns=self.columns)

    def add_score(self, img: str, metric_result):
        """
        Add a new row to the DataFrame with the given parameters.
        """
        new_row = {
            'img': img,
            f'{self.name_score}': metric_result,
        }
        # Append new row and reassign to self
        self.metric_df.loc[len(self.metric_df)] = new_row
    def add_column_value(self, column_name: str, value):
        """
        Adds a new column (if not present) and fills the next empty row with the given value.
        """
        # Check if the column exists, if not, create it
        if column_name not in self.metric_df.columns:
            self.metric_df[column_name] = pd.NA  # Initialize with empty values
        # Find the first empty row in the specified column
        empty_row_idx = self.metric_df[self.metric_df[column_name].isna()].index
        if len(empty_row_idx) > 0:
            # Assign the value to the first empty row in the specified column
            self.metric_df.at[empty_row_idx[0], column_name] = value
        else:
            raise  Exception(f"you must add the scores row before, The file cannot have more rows for the extra column than scores  len(rows)>=len({column_name})")
        
    def save_to_csv(self, file_to_save):
        """
        Save the DataFrame to a CSV file. Append to the file if it exists.
        """
        # Save the DataFrame, overriding the file if it already exists
        self.metric_df.to_csv(file_to_save, mode='w', header=True, index=False)
