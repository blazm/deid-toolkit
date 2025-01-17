import sys
import os
from tqdm import tqdm
sys.path.append(os.path.abspath("./root_dir/evaluation/identity_verification/AdaFace"))

from inference import load_pretrained_model, to_input
from face_alignment import align
import torch.nn as nn
import warnings
import utils as util
import numpy as np
import pickle
import torch
import torch.nn.functional as F
import shutil
from PIL import Image

EVALUATION_NAME= "adaface"

#load model
model = load_pretrained_model('ir_50').cuda()
model.eval()

# Function to save features to a file
def save_features(filepath, features):
    with open(filepath, 'wb') as f:
        pickle.dump(features.cpu(), f)

# Function to load features from a file
def load_features(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f).cuda()
# Function to compute the cosine similarity score
def compute_similarity(feature1, feature2):
    cosine_similarity = F.cosine_similarity(feature1, feature2)
    # Standardize the cosine similarity score to range [0, 1]
    similarity_score = (cosine_similarity + 1) / 2
    return similarity_score.item()
    
def get_features(image_path, features_dir): 
    image_name = os.path.basename(image_path)
    #save in temporary file
    feature_filepath = os.path.join(features_dir, f"{image_name}.pkl")
    # Check if the features have already been computed and saved
    if os.path.exists(feature_filepath):
        return load_features(feature_filepath)
    # Compute the features if not available
    aligned_rgb_img = Image.open(image_path).resize((112, 112), Image.Resampling.LANCZOS)
    #aligned_rgb_img = align.get_aligned_face(image_path)
    if aligned_rgb_img == None: 
        raise ValueError(f"{image_path} is not a facial image")

    bgr_input = to_input(aligned_rgb_img).cuda()
    with torch.no_grad():
        feature, _ = model(bgr_input)
    # Save the computed features for future use
    save_features(feature_filepath, feature)
    return feature

def main(): 
    #prevent warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    #get args
    args = util.read_args()
    path_to_aligned_images = args.aligned_path
    path_to_deidentified_images = args.deidentified_path
    path_to_genuine_pairs  = args.genuine_pairs_filepath
    path_to_impostor_pairs = args.impostor_pairs_filepath
    path_to_log = args.dir_to_log

    path_to_save = args.save_path
    dataset_name = args.dataset_name # = util.get_dataset_name_from_path(aligned_dataset_path)
    technique_name = args.technique_name # = util.get_technique_name_from_path(deid_dataset_path)
    deid_impostors = True # set this to true if you want to deidentify impostors as well
    metrics_df= util.Metrics(name_score="cossim")
    aligned_original_dataset = os.path.basename(path_to_aligned_images)
    # Create directories for features if they don't exist
    temp_features_original_dir= os.path.join(util.TEMP_DIR, EVALUATION_NAME,f"{aligned_original_dataset}", "original")
    temp_features_deid_dir = os.path.join(util.TEMP_DIR, EVALUATION_NAME,f"{dataset_name}_{technique_name}", "deid")
    os.makedirs(temp_features_original_dir, exist_ok=True)
    os.makedirs(temp_features_deid_dir, exist_ok=True)


    if path_to_impostor_pairs is None:
        print("No impostor pairs provided")
        return  
    if path_to_genuine_pairs is None:
        print("No genuine pairs provided")
        return  

      #get pairs from file
    genu_names_a, genu_ids_a, genu_names_b, genu_ids_b = util.read_pairs_file(path_to_genuine_pairs)
    impo_names_a, impo_ids_a, impo_names_b, impo_ids_b = util.read_pairs_file(path_to_impostor_pairs)
    
    names_a = genu_names_a + impo_names_a # images a are originals
    names_b = genu_names_b + impo_names_b # images b are deidentified
    # list of booleans, same size as names_a but all the genu_names_a positions will be true and all the impo_names_a positions will be false
    gt_labels_is_genuine = [True] * len(genu_names_a) + [False] * len(impo_names_a)
    for name_a, name_b,isGenuine in tqdm(zip(names_a, names_b,gt_labels_is_genuine), total=len(names_a), desc=f"adaface | {dataset_name}-{technique_name}"):
        img_a_path = os.path.abspath(os.path.join(path_to_aligned_images, name_a)) #the the aligned image file path
        if isGenuine or deid_impostors:
            img_b_path = os.path.abspath(os.path.join(path_to_deidentified_images, name_b)) #the deidentified image file path
        else:
            img_b_path = os.path.abspath(os.path.join(path_to_aligned_images, name_b))#take the image from the original paths
        if isGenuine or deid_impostors:
            img_c_path = os.path.abspath(os.path.join(path_to_aligned_images, name_b)) # genuine orig to see the original distribution
        if not os.path.exists(img_a_path):
            util.log(os.path.join(path_to_log,"adaface_optimized.txt"), 
                     f"({dataset_name}) The source images are not in {img_a_path} ")
            print(f"Source Images are not there! {img_a_path} ")
            continue 
        if not os.path.exists(img_b_path): # if any of the pipelines failed to detect faces
            util.log(os.path.join(path_to_log,"adaface_optimized.txt"), 
                     f"({technique_name}) The deidentified images are not in {img_b_path} ")
            print("Deid Images are not there! ", img_b_path)
            continue
        similarity_score = 0
        try:
            feature_a = get_features(img_a_path, temp_features_original_dir)
            feature_b = get_features(img_b_path, temp_features_deid_dir)
            if isGenuine or deid_impostors:
                feature_c = get_features(img_c_path, temp_features_original_dir)
        except ValueError as e:
            print(f"(Warning) {e} - Skip")
            continue
        
        # Compute similarity and add to the list
        similarity_score = compute_similarity(feature_a,feature_b)
        if isGenuine or deid_impostors:
            similarity_gens_score = compute_similarity(feature_a,feature_c)
        metrics_df.add_score(img=name_a, 
                             metric_result=similarity_score)
        metrics_df.add_column_value("img_b", name_b)
        metrics_df.add_column_value("ground_truth", isGenuine)
        if isGenuine or deid_impostors:
            metrics_df.add_column_value("cossim_originals", similarity_gens_score)
        else:
            metrics_df.add_column_value("cossim_originals", -1)
    metrics_df.save_to_csv(path_to_save)
    print(f"Adaface scores saved into {path_to_save}")
    shutil.rmtree(temp_features_deid_dir)
    return





if __name__ == "__main__":
    main()
