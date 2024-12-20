"""
Evaluation method: insight face
category: identity verification
"""
import argparse
import csv
import cv2
import sys
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image
from identity_verification.insightface.models.scrfd import SCRFD # recognition
from identity_verification.insightface.models.arcface_onnx import ArcFaceONNX # recognition


#import onnxruntime as ort
import os
import utils as util
from numpy import dot
from numpy.linalg import norm
#INSIGHT_MODEL_PATH = "./root_dir/evaluation/identity_verification/insightface/models/model.onnx" 
#INSIGHT_MODEL_PATH = "./root_dir/evaluation/identity_verification/insightface/models/mxnet_exported_R100.onnx" 
ROOT_DIR="./root_dir/evaluation/identity_verification/insightface/models/"
assets_dir = os.path.expanduser('~/.insightface/models/buffalo_l')
FACE_RECOGNITION_PATH=os.path.join(assets_dir, 'det_10g.onnx')
FACE_DETECTION_PATH=os.path.join(assets_dir, 'w600k_r50.onnx')



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
    img = cv2.resize(img, (112, 112)) 
    return img



def do_flip(data):
    for idx in range(data.shape[0]):
        data[idx, :, :] = np.fliplr(data[idx, :, :])

class FaceModel: 
    def __init__(self, detector_model_path,recognition_model_path):
        self.detector:SCRFD = SCRFD(detector_model_path)
        self.detector.prepare(0)
        
        self.recognition: ArcFaceONNX  = ArcFaceONNX(recognition_model_path)
        self.recognition.prepare(0)
    def get_feature(self, img):
        bboxes1, kpss1 = self.detector.autodetect(img, max_num=1)
        if bboxes1.shape[0]==0:
            return -1.0, "Face not found in Image-1"
        kps1 = kpss1[0]
        feat = self.recognition.get(img, kps1)
        return feat


#class FaceModel:
#    def __init__(self, recognition_model_path=""):
#        #self.handler = insightface.app.FaceAnalysis(allowed_modules=['recognition'], 
#        #                                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
#        # face detection
#        #model_pack_name = 'buffalo_l'
#        #self.detector = FaceAnalysis(name=model_pack_name)
#        #self.detector.prepare(ctx_id=0)
#        #face recognition
#        #self.handler = insightface.model_zoo.get_model(model_path)
#        #self.handler.prepare(ctx_id=0, model_path=model_path)
#        #self.handler = insightface.model_zoo.get_model(model_path)
#        #self.handler.prepare(ctx_id=0)
#        
#        #model_pack_name = 'buffalo_l'
#        self.handler = FaceAnalysis()
#        self.handler.prepare(ctx_id=0)
#        #self.detector = FaceAnalysis(allow_modules=["detection"]) # enable detection model only
#        #self.detector = insightface.model_zoo.get_model('retinaface_r50_v1')
#        #self.detector.prepare(ctx_id=0)
#
#        self.recognition = insightface.model_zoo.get_model(recognition_model_path)
#        self.recognition.prepare(ctx_id=0)
#    def get_feature(self,aligned):
#        self.recognition.get()
#    
#    
#    def get_feature(self, aligned):
#        faces = self.handler.get(aligned)
#        if len(faces) == 0:
#            print("No faces detected")
#            return None
#        print("face detected")
#        feat =faces
#        return feat
def normalize_tensor(im): 
    '''
    Normalize by mean and std.
    '''
    return (im - im.mean()) / im.std()

def main(): 
    args = util.read_args()
    result = util.MetricsBuilder()
    #get the mandatory args
    path_to_aligned_images = args.aligned_path
    path_to_deidentified_images = args.deidentified_path
    path_to_genuine_pairs  = args.genuine_pairs_filepath
    path_to_impostor_pairs = args.impostor_pairs_filepath
    output_file_name = util.get_output_filename("insight_face",path_to_aligned_images, path_to_deidentified_images)

    if path_to_impostor_pairs is None:
        print("No impostor pairs provided")
        return  
    if path_to_genuine_pairs is None:
        print("No genuine pairs provided")
        return  
    model = FaceModel(detector_model_path=FACE_DETECTION_PATH
                      ,recognition_model_path=FACE_RECOGNITION_PATH)
    
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
            predicted_scores.append(0.5) # so that the length of the array is equal to GT
            continue
        
        img_a = process_image(img_a_path, process_without_context = False)
        img_b = process_image(img_b_path, process_without_context = False)
        #img_a = get_image(img_a_path)
        #img_b = get_image(img_b_path)
        print(f"{img_a_path}")
        feat_a = model.get_feature(img_a)
        print(f"{img_b_path}")   
        feat_b = model.get_feature(img_b)

        #cos_sim = dot(a, b)/(norm(a)*norm(b))
        #cos_sim = dot(feat_a, feat_b)/(norm(feat_a)*norm(feat_b))
        #sim = np.dot(feat_a, feat_b)
        #predicted_scores.append(cos_sim)

    np.savetxt(output_file_name, predicted_scores)
    return result.add_metric("insightface","min", np.min(predicted_scores)).add_metric("insightface", "max",np.max(predicted_scores))


    
if __name__ == "__main__":
    main()
    #result, output, errors = util.with_no_prints(main)
    #result.add_error(errors)
    #result.add_output_message(output)
    #print(result.build())