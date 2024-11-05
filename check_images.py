import os
import math

def get_image_files(directory):
    # Get a set of image filenames in the directory (case-insensitive)
    return set(os.path.splitext(f.lower())[0] for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')))

def compare_directories(base_dir, target_dir):
    # Get image files in each directory
    base_images = get_image_files(base_dir)
    target_images = get_image_files(target_dir)
    
    # Find images that are in the base directory but not in the target diddrectory
    remaining_images = base_images - target_images
    remaining_count = len(remaining_images)
    
    # Calculate total images in base directory and percentage
    total_images = len(base_images)
    if total_images == 0:
        print("No images found in the base directory.")
        return

    remaining_percentage = (remaining_count / total_images) * 100
    
    # Display results
    print(f"Total images in base directory: {total_images}")
    print(f"Images remaining to appear in target directory: {remaining_count}")
    print(f"Percentage of images remaining: {remaining_percentage:.2f}%")

# Set your base and target directories
base_directory = "root_dir/datasets/aligned/arface"
#target_dirs = ['cleanir','AMT-GAN','blur','ksamenet','LeeCroft','pixelize']
target_dirs = ['AMT-GAN']
for dir_name in target_dirs:
    print(dir_name)
    if dir_name == 'AMT-GAN':
        for i in range(1,11):
            if i ==5:
                continue
            num_digits = int(math.log10(abs(i))) + 1
            if num_digits > 1:
                subdir = f'ref{i}'
            else:
                subdir = f'ref0{i}'
            print(subdir)
            target_directory = f"root_dir/datasets/{dir_name}/arface/{subdir}"
            compare_directories(base_directory, target_directory)
        continue
    target_directory = f"root_dir/datasets/{dir_name}/arface"
    compare_directories(base_directory, target_directory)
