import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
import imageio
import os

# weak pixelization size 8 or 16
def main(img_path, output_path, subs_size=32): # subs_size is fixed, does not change with k (plot straight line!)

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
        di = img[::subs_size, ::subs_size, :] # subsample by step
        di = resize(di, (h, w, ch), order=0, preserve_range=True, anti_aliasing=False) # resize to original size

        # Convert di to [0, 255] range if necessary
        if di.max() <= 1.0:
            di = (di * 255).astype(np.uint8)

        # Determine the file format based on the file extension
        file_extension = os.path.splitext(output_path)[1][1:]  # Get the extension without the dot

        # Save the pixelized image
        imageio.imwrite(output_path, di, format=file_extension)

        return di
    except Exception as e:
        print(f"An error occurred: {e}")

# if __name__ == "__main__":
#     img_path = '/home/matthieup/deid-toolkit/root_dir/datasets/aligned/fri/AjdaLampe.jpg'
#     output_path = os.path.join('/home/matthieup/deid-toolkit/root_dir/datasets/pixelelized/fri',"AjdaLampe.jpg")  # Change this to the desired output path
#     pixel_img = main(img_path, output_path)
#     if pixel_img is not None:
#         print(type(pixel_img))
#         plt.imshow(pixel_img)
#         plt.axis('off')
#         plt.show()
