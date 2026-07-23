import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
import imageio
import os
from tqdm import tqdm
import argparse
# weak pixelization size 8 or 16
def pixelize(img_path, output_path, subs_size=32): # subs_size is fixed, does not change with k (plot straight line!)

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
    
        h, w, ch = img.shape

        # Compute the exact target dimensions for downsample.
        # Using the same slicing pattern as `img[::subs_size, ::subs_size]` to keep
        # the output size deterministic and consistent with previous runs.
        target_h = (h + subs_size - 1) // subs_size
        target_w = (w + subs_size - 1) // subs_size

        # Downsample with area interpolation so each output pixel is a proper block
        # average of the corresponding region in the original image.
        di = resize(img, (target_h, target_w, ch), order=1, preserve_range=True, anti_aliasing=False)

        # Upscale back to the original size with nearest-neighbor interpolation.
        # Because we resized down to `target_h x target_w` (the same dimensions a
        # simple stride-subsample would produce), the subsequent nearest-neighbor
        # upscale maps each small pixel to a contiguous block without coordinate
        # drift, eliminating the black-bar artifact on the left/top edges.
        di = resize(di, (h, w, ch), order=0, preserve_range=True, anti_aliasing=False)

        # `preserve_range=True` keeps values in [0, 255] but as float.
        # Always cast to uint8 so imageio can write the file reliably.
        di = np.clip(di, 0, 255).astype(np.uint8)

        # Determine the file format based on the file extension
        file_extension = os.path.splitext(output_path)[1][1:]  # Get the extension without the dot

        # Save the pixelized image
        imageio.imwrite(output_path, di, format=file_extension)

        return di
    except Exception as e:
        print(f"An error occurred: {e}")

_TEST_SINGLE = int(os.environ.get("DEID_TEST_SINGLE", "0"))

def main(dir_path,save_dir):
        images = os.listdir(dir_path)
        dataset_name = os.path.basename(dir_path)
        for i, img in enumerate(tqdm(images, desc=f"Processing {dataset_name}")):
            if _TEST_SINGLE and i > 0:
                break
            input_path = os.path.join(dir_path, img)
            output_path = os.path.join(save_dir, img)
            try:
                pixelize(img_path=input_path, output_path=output_path)
            except Exception as e:
                print(f"Error processing image {img} with pixelization: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and picelize images.")
    parser.add_argument('dataset_path', type=str, help="Path to the dataset directory")
    parser.add_argument('dataset_save', type=str, help="Path to the save directory")
    parser.add_argument('--dataset_filetype', type=str, default='jpg', help="Filetype of the dataset images (default: jpg)")
    parser.add_argument('--dataset_newtype', type=str, default='jpg', help="Filetype for the anonymized images (default: jpg)")

    args = parser.parse_args()
    main(args.dataset_path, args.dataset_save)
