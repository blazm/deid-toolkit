

import sys
import os

sys.path.append(os.path.abspath("./root_dir/evaluation/identity_verification/AdaFace"))
from inference import load_pretrained_model, to_input
from face_alignment import align
import torch.nn as nn

#from identity_verification.AdaFace.inference import load_pretrained_model, to_input
import utils as util
import numpy as np
from PIL import Image
from numpy import dot
import torch
from numpy.linalg import norm

IMG_SIZE=112
def get_features(emb):
    emb =  np.sqrt(np.sum(emb*emb)+0.00001)
    emb /= norm
    return emb

def main(): 
    args = util.read_args()
    result = util.MetricsBuilder()
    #get the mandatory args
    path_to_aligned_images = args.aligned_path
    path_to_deidentified_images = args.deidentified_path
    path_to_genuine_pairs  = args.genuine_pairs_filepath
    path_to_impostor_pairs = args.impostor_pairs_filepath
    output_file_name = util.get_output_filename("adaface",path_to_aligned_images, path_to_deidentified_images)

    if path_to_impostor_pairs is None:
        print("No impostor pairs provided")
        return  
    if path_to_genuine_pairs is None:
        print("No genuine pairs provided")
        return  
    #load model
    model = load_pretrained_model('ir_50')
    

    #get pairs from file
    genu_names_a, genu_ids_a, genu_names_b, genu_ids_b = util.read_pairs_file(path_to_genuine_pairs)
    impo_names_a, impo_ids_a, impo_names_b, impo_ids_b = util.read_pairs_file(path_to_impostor_pairs)
    
    names_a = genu_names_a + impo_names_a # images a are originals
    names_b = genu_names_b + impo_names_b # images b are deidentified
    ids_a = genu_ids_a + impo_ids_a
    ids_b = genu_ids_b + impo_ids_b
    
    ground_truth_binary_labels = np.array([int(id_a == id_b) for id_a, id_b in zip(ids_a, ids_b)])
    predicted_scores = []

    for name_a, name_b, gt_label in zip(names_a, names_b, ground_truth_binary_labels): 
        img_a_path = os.path.abspath(os.path.join(path_to_aligned_images, name_a)) #the the aligned image file path
        img_b_path = os.path.abspath(os.path.join(path_to_deidentified_images, name_b)) #the deidentified image file path
        if not os.path.exists(img_a_path):
            print("Source Images are not there!")
            continue 
        if not os.path.exists(img_b_path): # if any of the pipelines failed to detect faces
            print("Deid Images are not there! ", img_b_path)
            continue
        
        aligned_rgb_img_a = align.get_aligned_face(img_a_path)
        aligned_rgb_img_b = align.get_aligned_face(img_b_path)
        bgr_input_a = to_input(aligned_rgb_img_a)
        bgr_input_b = to_input(aligned_rgb_img_b)
        feat_a, _ = model(bgr_input_a)
        feat_b, _ = model(bgr_input_b)
       
        # Detach features from the computation graph
        feat_a = feat_a.detach()
        feat_b = feat_b.detach()

        # Calculate cosine similarity
        cosim = nn.CosineSimilarity()
        cos_sim = cosim(feat_a, feat_b)

        predicted_scores.append((cos_sim.item()+1)/2)

    np.savetxt(output_file_name, predicted_scores )
    return result.add_metric("adaface","min", np.min(predicted_scores)).add_metric("adaface", "max",np.max(predicted_scores))


    
if __name__ == "__main__":
    result, output, errors = util.with_no_prints(main)
    result.add_output_message(str(output))
    result.add_error(str(errors))
    print(result.build())
