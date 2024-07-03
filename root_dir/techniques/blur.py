import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import imageio
import os
from tqdm import tqdm
import argparse

def blur(img_path, output_path, kernel_size=30):
    try:
        # Check if the directory of the output path exists and create it if it doesn't
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        img = plt.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at path: {img_path}")

        # Check if the image is in the range [0, 1] and convert to [0, 255] if necessary
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)

        sigma = kernel_size / 2

        di = gaussian_filter(img, sigma=(sigma, sigma, 0))

        # Convert di to [0, 255] range if necessary
        if di.max() <= 1.0:
            di = (di * 255).astype(np.uint8)

        # Determine the file format based on the file extension
        file_extension = os.path.splitext(output_path)[1][1:]  # Get the extension without the dot

        # Save the blurred image
        imageio.imwrite(output_path, di, format=file_extension)

        return di
    except Exception as e:
        print(f"An error occurred: {e}")

def main(dir_path,save_dir):
        images = os.listdir(dir_path)
        dataset_name = os.path.basename(dir_path)
        for img in tqdm(images, desc=f"Processing {dataset_name}"):
            input_path = os.path.join(dir_path, img)
            output_path = os.path.join(save_dir, img)
            try:
                blur(img_path=input_path, output_path=output_path)
            except Exception as e:
                print(f"Error processing image {img} with blur: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and anonymize images.")
    parser.add_argument('dataset_path', type=str, help="Path to the dataset directory")
    parser.add_argument('dataset_save', type=str, help="Path to the save directory")
    parser.add_argument('--dataset_filetype', type=str, default='jpg', help="Filetype of the dataset images (default: jpg)")
    parser.add_argument('--dataset_newtype', type=str, default='jpg', help="Filetype for the anonymized images (default: jpg)")

    args = parser.parse_args()
    main(args.dataset_path, args.dataset_save)