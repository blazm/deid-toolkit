import os
import subprocess
import numpy as np
import torch
import argparse
import select
from tqdm import tqdm 
from DeepPrivacy.deep_privacy import logger
from DeepPrivacy.deep_privacy.inference.deep_privacy_anonymizer import DeepPrivacyAnonymizer
from DeepPrivacy.deep_privacy.build import build_anonymizer, available_models
from face_detection.retinaface.detect import RetinaNetDetector
from face_detection import base, torch_utils
from face_detection.retinaface.config import cfg_mnet,cfg_re50
from face_detection.retinaface.models.retinaface import RetinaFace

class face_detection(base.Detector):
    def __init__(self, model: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model_dir = os.path.join("DeepPrivacy","deep_privacy","checkpoints")
        map_location = torch_utils.get_device()

        if model == "mobilenet":
            cfg = cfg_mnet
            state_dict = self.load_local_or_remote_state_dict(
                "RetinaFace_mobilenet025.pth",
                model_dir=model_dir,
                map_location=map_location
            )
        else:
            assert model == "resnet50"
            cfg = cfg_re50
            state_dict = self.load_local_or_remote_state_dict(
                "RetinaFace_ResNet50.pth",
                model_dir=model_dir,
                map_location=map_location
            )
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        net = RetinaFace(cfg=cfg)
        net.eval()
        net.load_state_dict(state_dict)
        self.cfg = cfg
        self.net = net.to(self.device)
        self.mean = np.array([104, 117, 123], dtype=np.float32)
        self.prior_box_cache = {}

    def load_local_or_remote_state_dict(self, filename, model_dir, map_location):
        local_path = os.path.join(model_dir, filename)
        if os.path.exists(local_path):
            state_dict = torch.load(local_path, map_location=map_location)
        else:
            raise FileNotFoundError(f"The model file {filename} does not exist in {model_dir}")
        return state_dict
    


def main(dataset_path, dataset_save, dataset_filetype='jpg', dataset_newtype='jpg'):
    img_names = [i for i in os.listdir(dataset_path) if dataset_filetype in i]
    img_paths = [os.path.join(dataset_path, i) for i in img_names]
    save_paths = [os.path.join(dataset_save, i.replace(dataset_filetype, dataset_newtype)) for i in img_names]
    anonymize_path = os.path.join('DeepPrivacy', 'anonymize.py')

    pbar = tqdm(total=len(img_paths), desc="Processing images", ncols=80)
    update_interval=int(np.round(2*len(img_names)/100))
    iteration_count = 0

    for img_path, save_path in zip(img_paths, save_paths):
        p = subprocess.Popen(['python', anonymize_path, '-s', img_path, '-t', dataset_save],
                             bufsize=2048, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        while True:
            reads = [p.stdout.fileno(), p.stderr.fileno()]
            ret = select.select(reads, [], [])
            for fd in ret[0]:
                if fd == p.stdout.fileno():
                    output = p.stdout.readline()
                    if output:
                        print(output.strip())
                if fd == p.stderr.fileno():
                    error = p.stderr.readline()
                    if error:
                        print(error.strip())

            if p.poll() is not None:
                break

        p.wait()

        if p.returncode == 0:
            print(f"Image: {img_path} processed OK.")
        else:
            print(f"Image: {img_path} FAILED.")
        
        iteration_count += 1
        if iteration_count >= update_interval:
            pbar.update(update_interval)
            iteration_count = 0

    if iteration_count > 0:
        pbar.update(iteration_count)

    pbar.close()

if __name__ == '__main__':
    RetinaNetDetector.__init__= face_detection.__init__
    parser = argparse.ArgumentParser(description="Process and anonymize images.")
    parser.add_argument('dataset_path', type=str, help="Path to the dataset directory")
    parser.add_argument('dataset_save', type=str, help="Path to the save directory")
    parser.add_argument('--dataset_filetype', type=str, default='jpg', help="Filetype of the dataset images (default: jpg)")
    parser.add_argument('--dataset_newtype', type=str, default='jpg', help="Filetype for the anonymized images (default: jpg)")

    args = parser.parse_args()
    main(args.dataset_path, args.dataset_save, args.dataset_filetype, args.dataset_newtype)

    
