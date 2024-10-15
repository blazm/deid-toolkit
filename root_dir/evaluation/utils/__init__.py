from PIL import Image
import numpy as np
import io
from contextlib import redirect_stdout, redirect_stderr
import json
import os
import argparse

import pandas as pd



"""
This file will provide usefull functions for the scripts that performs evaluation techniques.
"""
OUTPUT_FOLDER = "./deidtoolkit/root_dir/evaluation/output"
ROOT_FOLDER = "./root_dir/evaluation/"
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
    return aligned_path.split("/")[-1]
def get_technique_name_from_path(deidentified_path:str):
    """
    Args:
        deidentified_path (str): this is mandatory because the deidentified path must contain
        the {technique}/{dataset} in the path, index: -2

    Returns:
        str: name of the technique
    """
    return deidentified_path.split("/")[-2]

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


def with_no_prints(function, *args, **kwargs)->tuple:
    """
    This function will redirect the undesired prints to the returned variables, 
    to control the flow of the outputs from other three parties scripts

    Arguments: 
        function: the function to execte without undesired prints, 
        *args: the arguments of the function
    Returns:
        result: the returned value of the funcion
        output: the prints, without errors
        error: the error prints. 
    """
    f = io.StringIO()
    errors = io.StringIO()
    with redirect_stdout(f), redirect_stderr(errors):
        result = function(*args, **kwargs)
    output = f.getvalue()
    error_output = errors.getvalue()
    return result, output, error_output


class Metrics():
    metric_df = None
    def __init__(self, name_evaluation:str, name_dataset:str, name_technique:str, name_score:str):
        # Call the parent constructor (for DataFrame functionality)
        self.columns = ['aligned_path', 'deidentified_path', 'dataset', 'technique', 'score_name', 'metric_result']
        
        # Store the required parameters as class attributes
        self.name_dataset = name_dataset
        self.name_evaluation = name_evaluation
        self.name_technique = name_technique
        self.name_score = name_score

        self.metric_df = pd.DataFrame(columns=self.columns)

   

    def add_score(self, path_aligned: str, path_deidentified: str, metric_result):
        """
        Add a new row to the DataFrame with the given parameters.
        """
        new_row = {
            'dataset': self.name_dataset,
            'technique': self.name_technique,
            'aligned_path': path_aligned,
            'deidentified_path': path_deidentified,
            'score_name': self.name_score,
            'metric_result': metric_result
        }
        # Append new row and reassign to self
        self.metric_df.loc[len(self.metric_df)] = new_row
    def save_to_csv(self, file_to_save):
        """
        Save the DataFrame to a CSV file. Append to the file if it exists.
        """
        # Check if the file exists
        filepath = file_to_save
        if os.path.isfile(filepath):
            # Append without writing the header if the file already exists
            self.metric_df.to_csv(filepath, mode='a', header=False, index=False)
        else:
            # Write the file with the header if it doesn't exist
            self.metric_df.to_csv(filepath,header=True, mode='w', index=False)
