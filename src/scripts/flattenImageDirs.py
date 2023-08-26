import os.path

from tqdm import tqdm
import shutil
from cvusa import get_metadata, get_aerial

root = "/tf/CVUSA/"

with open('/tf/CVUSA/streetview_images.txt', 'r') as f:
    image_data = [(x.strip(),) + get_metadata(x.strip()) for x in f]

for data in tqdm(image_data):

    gnd_fname, lat, lon = data[0:3]
    sat_fname = os.path.join("streetview_aerial", get_aerial(lat, lon, 14))

    gnd_src = os.path.join(root, gnd_fname)
    sat_src = os.path.join(root, sat_fname)

    if os.path.exists(gnd_src) and os.path.exists(sat_src):

        sat_dest = os.path.join(root, "sat_streetview_train", sat_fname.split("/")[-1])
        shutil.copy2(sat_src, sat_dest)

        gnd_dest = os.path.join(root, "gnd_streetview_train", sat_fname.split("/")[-1])
        shutil.copy2(gnd_src, gnd_dest)

    elif not os.path.exists(gnd_src):
        print("gnd not found: ", os.path.join(root, gnd_fname))

    elif not os.path.exists(sat_src):
        print("sat not found: ", os.path.join(root, sat_fname))



