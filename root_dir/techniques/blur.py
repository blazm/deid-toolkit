import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import imageio
import os

def main(img_path, output_path, kernel_size=30):
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

# if __name__ == "__main__":
#     img_path = '/home/matthieup/deid-toolkit/root_dir/datasets/aligned/fri/AjdaLampe.jpg'
#     output_path = '/home/matthieup/deid-toolkit/root_dir/datasets/blurred/fri/AjdaLampe.jpg'  # Change this to the desired output path
#     blurred_image = main(img_path, output_path, kernel_size=30)
#     if blurred_image is not None:
#         print(type(blurred_image))
#         plt.imshow(blurred_image)
#         plt.axis('off')
#         plt.show()