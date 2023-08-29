import os
import sys
import cv2
import numpy as np
from tqdm import tqdm


# setting path
sys.path.append('../../src')
from utils import polar

# This script runs through a given directory, polar transforms all the images and then writes them
# to the destination.  Doing it once offline seems more efficient than for every training step.

src_dir = "/tf/CVUSA/sat_streetview_train"
dest_dir = "/tf/CVUSA/sat_streetview_train_polar/"

files = os.listdir(src_dir)

for file_name in tqdm(files):
    img_sat = cv2.imread(os.path.join(src_dir, file_name))
    img_sat = cv2.cvtColor(img_sat, cv2.COLOR_BGR2RGB)
    img_pol = np.multiply(polar(img_sat), 255)
    cv2.imwrite(os.path.join(dest_dir, file_name), img_pol)
