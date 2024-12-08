import os
import cv2
import numpy as np

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def enlarge_with_mirror(input_dir, output_dir):
    ensure_dir(output_dir)
    nb_img = len(os.listdir(input_dir))
    c=0
    for filename in os.listdir(input_dir):
        # print(filename)
        c += 1
        if c%1000==0:
            progression = np.round(100 * c / nb_img, 3)
            print(f"\n Progression: {progression}% \n")
        if filename.endswith(('.jpg', '.png', '.jpeg','.JPG')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            # Read the input image
            img = cv2.imread(input_path)
            if img is None:
                print(f"Could not read image {input_path}")
                continue

            mirrored_img = cv2.copyMakeBorder(img, 50, 50, 50, 50, cv2.BORDER_REFLECT)

            cv2.imwrite(output_path, mirrored_img)
            #print(f"Enlarged image saved as {output_path}")

if __name__ == "__main__":
    input_directory = 'root_dir/datasets/original/fdf/img'
    output_directory = 'root_dir/datasets/mirrored/fdf'
    enlarge_with_mirror(input_directory, output_directory)
