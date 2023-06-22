import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm


def random_crop(image: np.ndarray) -> np.ndarray:
    height = image.shape[0]
    width = image.shape[1]
    min_width = int(0.75 * width)
    min_height = int(0.75 * height)

    while True:
        # Generate random coordinates for the top-left corner of the crop
        top_left_x = np.random.randint(0, width - min_width + 1)
        top_left_y = np.random.randint(0, height - min_height + 1)

        bottom_right_x = top_left_x + min_width
        bottom_right_y = top_left_y + min_height

        # Extract the cropped region from the image
        cropped_image = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

        # Check if the cropped image meets the size requirement
        if cropped_image.shape[0] >= min_height and cropped_image.shape[1] >= min_width:
            return cropped_image


if __name__ == "__main__":

    src_dir_path = "/tf/CVUSA/sat_test"
    dest_dir_path = "/tf/CVUSA/sat_test_cropped"

    for file_name in tqdm(os.listdir(src_dir_path)):

        src = os.path.join(src_dir_path, file_name)

        image = cv2.imread(src)
        cropped_image = random_crop(image)

        dest = os.path.join(dest_dir_path, file_name)
        cv2.imwrite(dest, cropped_image)

