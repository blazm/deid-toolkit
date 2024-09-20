from PIL import Image
import numpy as np
import io
from contextlib import redirect_stdout, redirect_stderr
"""
This file will provide usefull functions for the scripts that performs evaluation techniques.
"""

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
    #TODO: fix the relative path to absolute path
    return f"./root_dir/evaluation/output/{metric_name}_{dataset_name}_{technique_name}.txt"

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

class MetricsBuilder:
    """
    This builder is the easiest way to build a result for every metric, 
    call the build() function to print the dicctionary if you're ready to print the result

    Usage: 
        result = MetricsBuilder()
        result.add_metric("mse", 0.423, mean)
            .add_metric("mse", 0.03, std)
            .add_metric("ssim", 0.23, mean)
            .add_metric("mssim", 0.303, mean)
            .build()
    """
    def __init__(self):
        self._output_result = {"result": [], "errors":[], "output_messages": []}
    def add_metric(self, metric_name:str, score_name="score",value="n/d"):
        """Adds or update a value"""
        for result in self._output_result['result']:
            if metric_name in result:
                result[metric_name][score_name] = value
                return self
        self._output_result['result'].append({metric_name: {score_name: value}})
        return self
    def add_error(self, message, error="Unexpected error"): 
        """If something goes wrong, you can add a error message"""
        self._output_result['errors'].append(message)
        return self
    def add_output_message(self,message):
        self._output_result["output_messages"].append(message)
        return self
    def build(self):
        """Call this function when you want to print the dictionary"""
        return self._output_result
    def reset(self):
        """restart"""
        self._output_result = {"result": [], "errors":[], "output_messages": []}
        return self


