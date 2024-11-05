import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
swinface_dir = os.path.join(current_dir, 'identity_verification', 'swinface')
sys.path.append(swinface_dir)
import numpy as np
import torch
from model import build_model
import utils as util
from tqdm import tqdm

PATH_TO_MODEL_WEIGHTS  = './root_dir/evaluation/identity_verification/swinface/checkpoint_step_79999_gpu_0.pt'


def process_image(image_path:str):
    img = cv2.imread(image_path) # original images have to be resampled to 112x112
    img = cv2.resize(img,  (112, 112)) 
    return img
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
    dict_checkpoint = torch.load(PATH_TO_MODEL_WEIGHTS)
    model.backbone.load_state_dict(dict_checkpoint["state_dict_backbone"])
    model.fam.load_state_dict(dict_checkpoint["state_dict_fam"])
    model.tss.load_state_dict(dict_checkpoint["state_dict_tss"])
    model.om.load_state_dict(dict_checkpoint["state_dict_om"])
    model.eval()
    #get pairs from file
    genu_names_a, genu_ids_a, genu_names_b, genu_ids_b = util.read_pairs_file(path_to_genuine_pairs)
    impo_names_a, impo_ids_a, impo_names_b, impo_ids_b = util.read_pairs_file(path_to_impostor_pairs)
    
    names_a = genu_names_a + impo_names_a # images a are originals
    names_b = genu_names_b + impo_names_b # images b are deidentified
    ids_a = genu_ids_a + impo_ids_a
    ids_b = genu_ids_b + impo_ids_b

    ground_truth_binary_labels = np.array([int(id_a == id_b) for id_a, id_b in zip(ids_a, ids_b)])
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
        img_a = process_image(img_a_path)
        img_b = process_image(img_b_path)
        output_a = model(img_a).numpy()
        output_b = model(img_b).numpy()



if __name__ == "__main__":
    main()
    
