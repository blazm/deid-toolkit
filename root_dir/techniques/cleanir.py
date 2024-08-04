import argparse
import os
from PIL import Image
from cleanir.cleanir.cleanir import Cleanir
from cleanir.cleanir.tools.crop_face import *
from tqdm import tqdm


def main(src_dir, dst_dir, nb_outputs):
    dsize = (64, 64)
    imgs = os.listdir(src_dir)
    num_imgs = len(imgs)
    
    pbar = tqdm(total=num_imgs, desc="Processing images", ncols=60)
    update_interval=int(np.round(2*num_imgs/100))
    count = 0
    iteration_count = 0
    
    for img in imgs:
        count += 1
        img_path = os.path.join(src_dir, img)
        img_name = img.split('.')[0]
        
        if os.path.exists(os.path.join(dst_dir, img_name + '.jpg')):
            continue
        
        try:
            image_orig, (boundings, face_img) = crop_face_from_file_b(img_path, dsize)
        except Exception as e:
            print("Error cropping face from image:", img_path, "| Error:", e)
            continue
        
        t, r, b, l = boundings
        orig_face_size = (image_orig[t:b, l:r].shape[:2][1], image_orig[t:b, l:r].shape[:2][0])
        deid = cleanir.get_deid_single_axis_func(face_img)
        step = 180 // nb_outputs
        
        for i in range(0, 181, step):
            try:
                de_img = deid(i)
                de_img = cv2.resize(de_img, orig_face_size)
                image_orig[t:b, l:r] = de_img
                im = Image.fromarray(image_orig)
                im.save(os.path.join(dst_dir, img_name + '.jpg'))
            except Exception as e:
                print("ERROR on image:", img_name, ", trying without detection. Error:", e)
                try:
                    face_img = face_recognition.load_image_file(img_path)
                    face_img = cv2.resize(face_img, dsize)
                    deid = cleanir.get_deid_single_axis_func(face_img)
                    de_img = deid(i)
                    image_orig = de_img
                    im = Image.fromarray(image_orig)
                    im.save(os.path.join(dst_dir, img_name + '.jpg'))
                except Exception as e:
                    print("Failed to process image:", img_name, "without detection. Error:", e)
    
        iteration_count += 1
        if iteration_count >= update_interval:
            pbar.update(update_interval)
            iteration_count = 0
    if iteration_count > 0:
        pbar.update(iteration_count)
    pbar.close()

if __name__ == '__main__':
    MODEL_PATH = os.path.join('cleanir','model')
    parser = argparse.ArgumentParser(description="Process and anonymize images.")
    parser.add_argument('dataset_path', type=str, help="Path to the dataset directory")
    parser.add_argument('dataset_save', type=str, help="Path to the save directory")
    parser.add_argument('--dataset_filetype', type=str, default='jpg', help="Filetype of the dataset images (default: jpg)")
    parser.add_argument('--dataset_newtype', type=str, default='jpg', help="Filetype for the anonymized images (default: jpg)")

    print("Creating Cleanir.")
    cleanir = Cleanir()
    print("loading models...")
    cleanir.load_models(MODEL_PATH)
    print("Models loaded.")

    args = parser.parse_args()
    main(src_dir=args.dataset_path, dst_dir=args.dataset_save,nb_outputs=1)