from email.mime import image
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
swinface_dir = os.path.join(current_dir, 'identity_verification', 'swinface')
sys.path.append(swinface_dir)
import numpy as np
import torch
import torch.nn.functional as F

#import swinface
from identity_verification.swinface import build_model
import utils as util
from tqdm import tqdm
import cv2
import pickle

EVALUATION_NAME= "swinface"
PATH_TO_MODEL_WEIGHTS  = './root_dir/evaluation/identity_verification/swinface/checkpoint_step_79999_gpu_0.pt'
# Function to save features to a file
def save_features(filepath, features):
    with open(filepath, 'wb') as f:
        features_dict_cpu = {key: value.cpu() for key, value in features.items()}
        pickle.dump(features_dict_cpu, f)

# Function to load features from a file
def load_features(filepath):
    with open(filepath, 'rb') as f:
        loaded_features=pickle.load(f)
        features_cuda = {key: value.cpu() for key, value in loaded_features.items()}
        return features_cuda
def get_features(image_path, features_dir, model):
    image_name = os.path.basename(image_path)

    feature_filepath = os.path.join(features_dir, f"{image_name}.pkl")
    if os.path.exists(feature_filepath):
        return load_features(feature_filepath)
    img_a = process_image(image_path)
    features = model(img_a)
    save_features(feature_filepath, features)
    return features

def process_image(image_path:str):
    img = cv2.imread(image_path) # original images have to be resampled to 112x112
    img = cv2.resize(img,  (112, 112)) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    return img
def compute_similarity(feature1, feature2):
    cosine_similarity = F.cosine_similarity(feature1, feature2)
    # Standardize the cosine similarity score to range [0, 1]
    similarity_score = (cosine_similarity + 1) / 2
    return similarity_score.item()
class SwinFaceCfg:
    network = "swin_t"
    fam_kernel_size=3
    fam_in_chans=2112
    fam_conv_shared=False
    fam_conv_mode="split"
    fam_channel_attention="CBAM"
    fam_spatial_attention=None
    fam_pooling="max"
    fam_la_num_list=[2 for j in range(11)]
    fam_feature="all"
    fam = "3x3_2112_F_s_C_N_max"
    embedding_size = 512
def main():
    args = util.read_args()
    path_to_aligned_images = args.aligned_path
    path_to_deidentified_images = args.deidentified_path
    path_to_genuine_pairs  = args.genuine_pairs_filepath
    path_to_impostor_pairs = args.impostor_pairs_filepath
    path_to_save = args.save_path
    path_to_log = args.dir_to_log
    dataset_name = util.get_dataset_name_from_path(path_to_aligned_images)
    technique_name = util.get_technique_name_from_path(path_to_deidentified_images)
    # Create directories for features if they don't exist
    temp_features_original_dir= os.path.join(util.TEMP_DIR, EVALUATION_NAME,f"{dataset_name}_{technique_name}", "original")
    temp_features_deid_dir = os.path.join(util.TEMP_DIR, EVALUATION_NAME,f"{dataset_name}_{technique_name}", "deid")
    os.makedirs(temp_features_original_dir, exist_ok=True)
    os.makedirs(temp_features_deid_dir, exist_ok=True)
    
    metric_df= util.Metrics(name_score="cossim")

    #output_file_name = util.get_output_filename("vgg",path_to_aligned_images, path_to_deidentified_images)

    if path_to_impostor_pairs is None:
        print("No impostor pairs provided")
        return  
    if path_to_genuine_pairs is None:
        print("No genuine pairs provided")
        return  
    #initialize the model 
    cfg = SwinFaceCfg()
    model = build_model(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dict_checkpoint = torch.load(PATH_TO_MODEL_WEIGHTS,map_location=device)
    model.backbone.load_state_dict(dict_checkpoint["state_dict_backbone"])
    model.fam.load_state_dict(dict_checkpoint["state_dict_fam"])
    model.tss.load_state_dict(dict_checkpoint["state_dict_tss"])
    model.om.load_state_dict(dict_checkpoint["state_dict_om"])
    model.eval()
    metrics_df= util.Metrics(name_score="cossim")

    #get pairs from file
    genu_names_a, genu_ids_a, genu_names_b, genu_ids_b = util.read_pairs_file(path_to_genuine_pairs)
    impo_names_a, impo_ids_a, impo_names_b, impo_ids_b = util.read_pairs_file(path_to_impostor_pairs)
    
    names_a = genu_names_a + impo_names_a # images a are originals
    names_b = genu_names_b + impo_names_b # images b are deidentified
    ids_a = genu_ids_a + impo_ids_a
    ids_b = genu_ids_b + impo_ids_b

    ground_truth_binary_labels = np.array([int(id_a == id_b) for id_a, id_b in zip(ids_a, ids_b)])
    for name_a, name_b, gt_label in tqdm(zip(names_a, names_b, ground_truth_binary_labels), total=len(names_a), desc=f"swinface | {dataset_name}-{technique_name}"):
        img_a_path = os.path.join(path_to_aligned_images, name_a) #the the aligned image file path
        img_b_path = os.path.join(path_to_deidentified_images, name_b) #the deidentified image file path
        if not os.path.exists(img_a_path):
            util.log(os.path.join(path_to_log,"swinface.txt"), 
                    f"({dataset_name}) The source images are not in {img_a_path} ")
            print(f"{img_a_path} does not exist")
            continue
        if not  os.path.exists(img_b_path):
            util.log(os.path.join(path_to_log,"swinface.txt"), 
                    f"({technique_name}) The deidentified images are not in {img_b_path} ")
            print(f"{img_b_path} does not exist")
            continue    
        #check if the features exist
        #if exist in path load
        #else compute and save 

        
        features_a = get_features(img_a_path, temp_features_original_dir, model=model)
        features_b = get_features(img_b_path, temp_features_deid_dir, model=model )

        similarity_score = compute_similarity(features_a["Recognition"], features_b["Recognition"])
        #we can add more about the output
        metrics_df.add_score(img=name_a, 
                             metric_result=similarity_score)
        metrics_df.add_column_value("img_b", name_b)
        metrics_df.add_column_value("ground_truth", gt_label)

        #for each in output_a.keys():
        #    print(each, "\t", output_a[each][0].detach().numpy())

    metrics_df.save_to_csv(path_to_save)
    print(f"swinface scores saved into {path_to_save}")


if __name__ == "__main__":
    main()
    
