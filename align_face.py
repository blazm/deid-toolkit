"""
brief: face alignment with FFHQ method (https://github.com/NVlabs/ffhq-dataset)
author: lzhbrian (https://lzhbrian.me)
date: 2020.1.5
note: code is heavily borrowed from
    https://github.com/NVlabs/ffhq-dataset
    http://dlib.net/face_landmark_detection.py.html

requirements:
    apt install cmake
    conda install Pillow numpy scipy
    pip install dlib
    # download face landmark model from:
    # http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
"""

import numpy as np
import PIL
import PIL.Image
import os
import scipy
import scipy.ndimage
import dlib
from pathlib import Path
import cv2

root = Path()
print("root", root.resolve().as_posix())

# download model from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
predictor = dlib.shape_predictor(
    "root_dir/preprocess/shape_predictor_68_face_landmarks.dat"
)


def get_landmark(filepath):
    """get landmark with dlib
    :return: np.array shape=(68, 2)
    """
    detector = dlib.get_frontal_face_detector()

    # try:
    #     img = dlib.load_rgb_image(filepath)
    #     # print("len =",len(img.shape), "\n shape:",img.shape)
    # except RuntimeError:  # PPM files not supported by dlib
    #     import cv2
    img = cv2.imread(filepath)
    print(img.shape)
    dets = detector(img, 1)

    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        print(
            "Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()
            )
        )
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        print("Part 0: {}, Part 1: {} ...".format(shape.part(0), shape.part(1)))

    t = list(shape.parts())
    a = []
    for tt in t:
        a.append([tt.x, tt.y])
    lm = np.array(a)
    # lm is a shape=(68,2) np.array
    return lm


def align_face(filepath):
    """
    :param filepath: str
    :return: PIL Image
    """

    lm = get_landmark(filepath)

    lm_chin = lm[0:17]  # left-right
    lm_eyebrow_left = lm[17:22]  # left-right
    lm_eyebrow_right = lm[22:27]  # left-right
    lm_nose = lm[27:31]  # top-down
    lm_nostrils = lm[31:36]  # top-down
    lm_eye_left = lm[36:42]  # left-clockwise
    lm_eye_right = lm[42:48]  # left-clockwise
    lm_mouth_outer = lm[48:60]  # left-clockwise
    lm_mouth_inner = lm[60:68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # read image
    img = PIL.Image.open(filepath)

    output_size = 1024
    transform_size = 4096
    enable_padding = True

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (
            int(np.rint(float(img.size[0]) / shrink)),
            int(np.rint(float(img.size[1]) / shrink)),
        )
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (
        int(np.floor(min(quad[:, 0]))),
        int(np.floor(min(quad[:, 1]))),
        int(np.ceil(max(quad[:, 0]))),
        int(np.ceil(max(quad[:, 1]))),
    )
    crop = (
        max(crop[0] - border, 0),
        max(crop[1] - border, 0),
        min(crop[2] + border, img.size[0]),
        min(crop[3] + border, img.size[1]),
    )
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (
        int(np.floor(min(quad[:, 0]))),
        int(np.floor(min(quad[:, 1]))),
        int(np.ceil(max(quad[:, 0]))),
        int(np.ceil(max(quad[:, 1]))),
    )
    pad = (
        max(-pad[0] + border, 0),
        max(-pad[1] + border, 0),
        max(pad[2] - img.size[0] + border, 0),
        max(pad[3] - img.size[1] + border, 0),
    )
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(
            np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), "reflect"
        )
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(
            1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
            1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]),
        )
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(
            mask * 3.0 + 1.0, 0.0, 1.0
        )
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), "RGB")
        quad += pad[:2]

    # Transform.
    img = img.transform(
        (transform_size, transform_size),
        PIL.Image.QUAD,
        (quad + 0.5).flatten(),
        PIL.Image.BILINEAR,
    )
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

    # Save aligned image.
    return img


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if os.path.exists(directory):
        print(f"file:{file_path} exists")
    if not os.path.exists(directory):
        os.makedirs(directory)


def main(img_paths=None, save_paths=None, dataset_path=None, dataset_save_path=None):
    dataset_filetype = "jpg"
    dataset_filetype_2 = "png"
    dataset_newtype = "jpg"
    if dataset_path is not None:
        img_names = [
            i
            for i in os.listdir(dataset_path)
            if (dataset_filetype in i or dataset_filetype_2 in i)
        ]  # change into jpg
        img_paths = [os.path.join(dataset_path, i) for i in img_names]
        save_paths = [
            os.path.join(
                dataset_save_path, os.path.splitext(i)[0] + "." + dataset_newtype
            )
            for i in img_names
        ]
        # print(save_paths)
    count = 0
    for img_path, save_path in zip(img_paths, save_paths):
        # print(f"image_path:{img_path}")
        # print(f"save_path:{save_path}")
        try:
            img = align_face(img_path)
            # output_size=112
            # img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)
        except Exception as e:
            print(f"\n  {e} \n")
            # alignment can fail if face not detected
            """
            print("ERROR: Image: ", img_path)
            img = PIL.Image.open(img_path) # just read, resize and save as is
            output_size=112
            img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)
            #f = open(save_path,"w+") # this created empty files
            #f.close()
            """
            continue  # dont save, if not aligned
        # check if save_path exists and create directories
        # print("\n  BEFORE ENSURE \n")
        ensure_dir(save_path)
        # print("\n  AFTER ENSURE \n")

        # from PIL import Image
        # pil_img = Image.fromarray(img)
        # print("\n BEFORE SAVING \n")
        img.save(save_path)
        print("Image saved as: ", save_path)
        count += 1
        if count % 10 == 0:
            print(f"\n preprocessed image: {100*count/len(img_paths)} % \n")


# if __name__ == "__main__":
#     main(
#         dataset_path="root_dir\datasets\original\\fri",
#         dataset_save_path="root_dir/datasets/aligned/fri",
#     )

#     #'''
#     # example for array of images (LRV group)
#     img_paths = ['FRI(demo dataset)\AjdaLampe.jpg']#["mike.jpg", 'BorutBatagelj.jpg', 'MatejVitek.jpg', 'FrancSolina.jpg', 'ZigaEmersic.jpg']# ['blaz_meden_small.png', 'PeterPeer.jpg', 'VitoStruc.jpg']
#     save_paths = ['FRI(demo dataset)\\test_aligned\\aligned_AjdaLampe.jpg']#["aligned_mike.jpg", 'aligned_BorutBatagelj.jpg', 'aligned_MatejVitek.jpg', 'aligned_FrancSolina.jpg', 'aligned_ZigaEmersic.jpg']
#     #print(f"image_paths:{img_paths}")
#     #print(f"save_paths:{save_paths}")

#     # img_paths = ['TeaBrasanac.jpg', 'NikoGamulin.jpg', 'MajaMurnik.jpg', 'AlesJaklic.jpg', 'AjdaLampe.jpg', 'thrall5.jpg']
#     # save_paths = ['aligned_TeaBrasanac.jpg', 'aligned_NikoGamulin.jpg', 'aligned_MajaMurnik.jpg', 'aligned_AlesJaklic.jpg', 'aligned_AjdaLampe.jpg', 'aligned_thrall5.jpg']

#     # img_paths = ['DiegoSusanj.jpg']
#     # save_paths = ['aligned_DiegoSusanj.jpg']

#     # img_paths = ['MarkoFirm.png', 'AljosaBesednjak.png', 'AndrejSeruga.png']
#     # save_paths = ['aligned_MarkoFirm.jpg', 'aligned_AljosaBesednjak.jpg', 'aligned_AndrejSeruga.jpg']

#     # dataset_path = './datasets/lrv/'
#     # img_paths = [os.path.join(dataset_path, i) for i in img_paths]
#     # dataset_save = './datasets/lrv_aligned/'
#     # save_paths = [os.path.join(dataset_save, i) for i in save_paths]

#     #'''
#     #'''
#     # example for whole dataset
#     FRI_dataset_path = 'FRI(demo dataset)'
#     FRI_dataset_save = 'datasets/test_aligned'
#     # dataset_path = 'datasets/rafd_k-Same-Net_original'
#     # dataset_save = 'datasets/rafd_k-Same-Net_deidentified'
#     # dataset_path = '../DAN/datasets/AffectNet/val_set/images'
#     # dataset_save = '../DAN/datasets/AffectNet/val_set/aligned_images'
#     # dataset_path = '/media/blaz/Storage/datasets/face datasets/RaFD/RafD_45_135'
#     # dataset_save = '/media/blaz/Storage/datasets/face datasets/RaFD/RafD_45_135_aligned'
#     # dataset_path = '../DAN/datasets/AffectNet/train_set/images'
#     # dataset_save = '../DAN/datasets/AffectNet/train_set/aligned_images'
#     # dataset_path = '../DAN/datasets/AffectNet/val_set/images'
#     # dataset_save = '../DAN/datasets/AffectNet/val_set/aligned_images'

#     # dataset_path = '/media/blaz/Storage/datasets/results/manfred/GNN_Lee_Croft_inferences/500'
#     # dataset_save = '/media/blaz/Storage/datasets/results/manfred/GNN_Lee_Croft_inferences/500_aligned'
#     # dataset_path = '/media/blaz/Storage/datasets/results/manfred/inferences_CNN_AnonFACES/k19_inferences' # 2,3,5,7,13,19
#     # dataset_save = '/media/blaz/Storage/datasets/results/manfred/inferences_CNN_AnonFACES/k19_aligned'

#     # dataset_path = '../DAN/datasets/AffectNet/train_set/images'
#     # dataset_save = '../DAN/datasets/AffectNet/train_set/small_only_aligned_images'
#     # #dataset_path = '../DAN/datasets/AffectNet/train_set/small_images'
#     # dataset_path = '../DAN/datasets/AffectNet/val_set/images'
#     # dataset_save = '../DAN/datasets/AffectNet/val_set/small_only_aligned_images'

#     # #dataset_path = 'datasets/xm2vts_k-same-net_orig'
#     # #dataset_save = 'datasets/xm2vts_k-same-net'
#     dataset_filetype = 'jpg'
#     dataset_newtype = 'jpg'

#     # dataset_path = '/media/blaz/Storage/datasets/face datasets/emotion/RaFD2 - Radboud Faces Database/RafDDownload-90_45_135'
#     # dataset_save = '/media/blaz/Storage/datasets/face datasets/emotion/RaFD2 - Radboud Faces Database/RafDDownload-90_45_135_aligned'

#     # dataset_path = '/home/blaz/Downloads/AffectNet_Sample_uncropped/val_class_joined'
#     # dataset_save = '/home/blaz/Downloads/AffectNet_Sample_uncropped/val_class_aligned'

#     # dataset_path = '/home/blaz/datasets/affectnet/affectnet_k-Same-Net_src'
#     # dataset_save = '/home/blaz/datasets/affectnet/affectnet_k-Same-Net'

#     # dataset_path = '/home/blaz/datasets/celeba/celeba_k-Same-Net_src'
#     # dataset_save = '/home/blaz/datasets/celeba/celeba_k-Same-Net'

#     # dataset_path = '/home/blaz/datasets/rafd45/rafd45_k-Same-Net_src'
#     # dataset_save = '/home/blaz/datasets/rafd45/rafd45_k-Same-Net'

#     # dataset_path = '/home/blaz/datasets/xm2vts/xm2vts_k-Same-Net_src'
#     # dataset_save = '/home/blaz/datasets/xm2vts/xm2vts_k-Same-Net'

#     # dataset_path = '/home/blaz/datasets/rafd45/rafd45_Croft_et_al_v2'
#     # dataset_save = '/home/blaz/datasets/rafd45/Croft et al. v2'

#     # dataset_path = '/home/blaz/datasets/rafd45/rafd45_Croft_et_al_v2_ep1500_NO_IDS'
#     # dataset_save = '/home/blaz/datasets/rafd45/Croft et al. v2_NO_IDS'


#     #dataset_path = '/home/blaz/datasets/celeba/celeba_Croft_et_al'
#     #dataset_save = '/home/blaz/datasets/celeba/Croft et al. v2'
#     #dataset_path = '/home/blaz/datasets/affectnet/affectnet_Croft_et_al_src'
#     #dataset_save = '/home/blaz/datasets/affectnet/affectnet_Croft_et_al'
#     #dataset_path = '/home/blaz/datasets/xm2vts/xm2vts_Croft_et_al_src'
#     #dataset_save = '/home/blaz/datasets/xm2vts/xm2vts_Croft_et_al'

#     # img_names = [i for i in os.listdir(FRI_dataset_path) if dataset_filetype in i] # change ppm into jpg
#     # img_paths = [os.path.join(FRI_dataset_path, i) for i in img_names]
#     # save_paths = [os.path.join(FRI_dataset_save, i.replace(dataset_filetype, dataset_newtype)) for i in img_names]
#     #'''
#     #print(img_paths)
#     #print(save_paths)

#     #exit()

#     # pngs are not properly generated due to 4 channels (RGBA)
#     # for img_path, save_path in zip(img_paths, save_paths):
#     #     #img_path = 'blaz_meden_small.png'
#     #     print(f"image_path:{img_path}")
#     #     print(f"save_path:{save_path}")
#     #     try:
#     #         img = align_face(img_path)
#     #         #output_size=112
#     #         #img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)
#     #     except:
#     #         # alignment can fail if face not detected
#     #         '''
#     #         print("ERROR: Image: ", img_path)
#     #         img = PIL.Image.open(img_path) # just read, resize and save as is
#     #         output_size=112
#     #         img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)
#     #         #f = open(save_path,"w+") # this created empty files
#     #         #f.close()
#     #         '''
#     #         continue # dont save, if not aligned
#     #     #check if save_path exists and create directories
#     #     ensure_dir(save_path)

#     #     #from PIL import Image
#     #     #pil_img = Image.fromarray(img)
#     #     img.save(save_path)
#     #     print("Image saved as: ", save_path)
