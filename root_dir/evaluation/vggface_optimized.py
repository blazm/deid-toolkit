import sys
sys.path.insert(1, './root_dir/evaluation/identity_verification/vgg-face.pytorch/models')
import torch
import vgg_face
import utils as util
import numpy as np
from torch import nn
import os
from torchvision import transforms
import cv2
from tqdm import tqdm
import pickle


PATH_TO_MODEL_WEIGHTS  = './root_dir/evaluation/identity_verification/vgg-face.pytorch/pretrained/vgg_face_torch/VGG_FACE.t7'
EVALUATION_NAME= "vggface"


def crop_context(image, percent=0.1367): # same percentage as in InsightFace
    #print("Crop context image shape: ", image.shape)
    h, w, ch = image.shape 
    nw, nh = (int)(w * (1.0-(2*percent))), (int)(h * (1.0-(2*percent)))
    ow, oh = (int)((w - nw) /2.0), (int)(((h - nh ) /2.0) ) #+ ((h*percent)*0.75)) # moves face area down by 26px, (34px + 26px = 60px in total)
    #print(nw, nh, ow, oh)
    cropped_image = image[oh:oh+nh, ow:ow+nw, :]
    #print("Cropped image shape: ", cropped_image.shape)
    return cropped_image
def process_image(image_path:str, process_without_context=True):
    img = cv2.imread(image_path) # original images have to be resampled to 112x112
    if process_without_context:
        img = crop_context(img)
    img = cv2.resize(img, (224, 224)) 
    return img
# Function to save features to a file
def save_features(filepath, features):
    with open(filepath, 'wb') as f:
        pickle.dump(features.cpu(), f)

# Function to load features from a file
def load_features(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f).cuda()

def main():
    args = util.read_args()
    #get the mandatory args
    path_to_aligned_images = args.aligned_path
    path_to_deidentified_images = args.deidentified_path
    path_to_genuine_pairs  = args.genuine_pairs_filepath
    path_to_impostor_pairs = args.impostor_pairs_filepath
    path_to_save = args.save_path
    path_to_log = args.dir_to_log

    dataset_name = util.get_dataset_name_from_path(path_to_aligned_images)
    technique_name = util.get_technique_name_from_path(path_to_deidentified_images)
    metric_df= util.Metrics(name_score="cossim")

    #output_file_name = util.get_output_filename("vgg",path_to_aligned_images, path_to_deidentified_images)
    # Create directories for features if they don't exist
    temp_features_original_dir= os.path.join(util.TEMP_DIR, EVALUATION_NAME,f"{dataset_name}_{technique_name}", "original")
    temp_features_deid_dir = os.path.join(util.TEMP_DIR, EVALUATION_NAME,f"{dataset_name}_{technique_name}", "deid")
    os.makedirs(temp_features_original_dir, exist_ok=True)
    os.makedirs(temp_features_deid_dir, exist_ok=True)

    if path_to_impostor_pairs is None:
        print("No impostor pairs provided")
        return  
    if path_to_genuine_pairs is None:
        print("No genuine pairs provided")
        return  
    # Initialize the model
    device = torch.device("cuda:0")  # Uncomment this to run on GPU
    vgg_model = vgg_face.VGG_16().float().cuda() # .double()
    vgg_model.load_weights(path=PATH_TO_MODEL_WEIGHTS)
    vgg_model.eval()
    vgg_normalization_tensor = torch.Tensor(np.array([129.1863, 104.7624, 93.5940])).float().view(1, 3, 1, 1).to(device) # .double()

    #get pairs from file
    genu_names_a, genu_ids_a, genu_names_b, genu_ids_b = util.read_pairs_file(path_to_genuine_pairs)
    impo_names_a, impo_ids_a, impo_names_b, impo_ids_b = util.read_pairs_file(path_to_impostor_pairs)
    
    names_a = genu_names_a + impo_names_a # images a are originals
    names_b = genu_names_b + impo_names_b # images b are deidentified
    ids_a = genu_ids_a + impo_ids_a
    ids_b = genu_ids_b + impo_ids_b

    ground_truth_binary_labels = np.array([int(id_a == id_b) for id_a, id_b in zip(ids_a, ids_b)])
    vgg_predicted_scores = []
    
    for name_a, name_b, gt_label in tqdm(zip(names_a, names_b, ground_truth_binary_labels), total=len(names_a), desc=f"vggface | {dataset_name}-{technique_name}"):
        img_a_path = os.path.join(path_to_aligned_images, name_a) #the the aligned image file path
        img_b_path = os.path.join(path_to_deidentified_images, name_b) #the deidentified image file path
        if not os.path.exists(img_a_path):
            util.log(os.path.join(path_to_log,"vggface.txt"), 
                    f"({dataset_name}) The source images are not in {img_a_path} ")
            print(f"{img_a_path} does not exist")
            continue
        if not  os.path.exists(img_b_path):
            util.log(os.path.join(path_to_log,"vggface.txt"), 
                    f"({technique_name}) The deidentified images are not in {img_b_path} ")
            print(f"{img_b_path} does not exist")
            continue
        
        feature_filepath_a = os.path.join(temp_features_original_dir, f"{name_a}.pkl")
        feature_filepath_b = os.path.join(temp_features_deid_dir, f"{name_b}.pkl")
        #get features for a
        if os.path.exists(feature_filepath_a):
            features_a = load_features(feature_filepath_a)
        else:
            img_a = process_image(img_a_path, process_without_context = True)
            vgg_img_a = torch.Tensor(img_a).permute(2, 0, 1).float().to(device)
            features_a = vgg_model.forward_features_fc7(vgg_img_a - vgg_normalization_tensor).flatten()
        #get features for b
        if os.path.exists(feature_filepath_b):
            features_b = load_features(feature_filepath_b)
        else:
            img_b = process_image(img_b_path, process_without_context = True)
            vgg_img_b = torch.Tensor(img_b).permute(2, 0, 1).float().to(device)
            features_b = vgg_model.forward_features_fc7(vgg_img_b - vgg_normalization_tensor).flatten()
        
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        cos_score = cos(features_a.flatten(), features_b.flatten()).detach().cpu().numpy() #[0]
        #store results in a variable
        vgg_predicted_scores.append(cos_score)
        metric_df.add_score(name_a, cos_score)
        metric_df.add_column_value("ground_truth", gt_label)


    
    vgg_predicted_scores = np.array(vgg_predicted_scores)
    print("MIN:", np.min(vgg_predicted_scores), " MAX: ", np.max(vgg_predicted_scores))
    metric_df.save_to_csv(path_to_save)
    print(f"vggface saved in {path_to_save}")
    #np.savetxt(output_file_name, vgg_predicted_scores)
    return


if __name__ == "__main__":
    main()
