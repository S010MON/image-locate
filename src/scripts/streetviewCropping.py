import os
import shutil
import cv2
from tqdm import tqdm

num_sections = 4
src_dir_path_gnd = "/tf/CVUSA/CVPR_subset/streetview"
dest_dir_path_gnd = "/tf/CVUSA/CVPR_subset/gnd_crops"
src_dir_path_sat = "/tf/CVUSA/CVPR_subset/bingmap/18/"
dest_dir_path_sat = "/tf/CVUSA/CVPR_subset/sat_crops/"
file_names = os.listdir(src_dir_path_gnd)


for file_name_jpg in tqdm(file_names):

    # Load the panoramic image using OpenCV
    source_path = os.path.join(src_dir_path_gnd, file_name_jpg)
    panoramic_image = cv2.imread(source_path)

    # Calculate the width of each section
    image_width = panoramic_image.shape[1]
    section_width = image_width // num_sections

    file_name = file_name_jpg.split(".")[0]

    # Iterate over each section
    for i in range(num_sections):
        # Calculate the starting and ending indices of the section
        start_x = i * section_width
        end_x = (i + 1) * section_width

        # Extract the section from the panoramic image
        section = panoramic_image[:, start_x:end_x, :]

        # Save the section to a file with an appended index
        dest_gnd = os.path.join(dest_dir_path_gnd, f"{file_name}_{i + 1}.jpg")
        cv2.imwrite(dest_gnd, section)

        src_sat = os.path.join(src_dir_path_sat, file_name_jpg)
        dest_sat = os.path.join(dest_dir_path_sat, f"{file_name}_{i + 1}.jpg")
        shutil.copy2(src_sat, dest_sat)
