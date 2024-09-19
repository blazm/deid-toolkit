from PIL import Image
import numpy as np

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
        img1: Pillow image reference image.
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
