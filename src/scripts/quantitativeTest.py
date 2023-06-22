import os
import sys
import numpy as np
from tqdm import tqdm
from pathlib import Path
from scipy.spatial.distance import cdist


os.environ[
    'TF_CPP_MIN_LOG_LEVEL'] = '2'  # Bug workaround source: https://stackoverflow.com/questions/38073432/how-to-suppress-verbose-tensorflow-logging
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # Allows the importing of parent modules
from models import SiameseModel
from utils import load_and_preprocess_img
from metrics import recall_at_k

model = SiameseModel(base_network='vgg16', netvlad=True)
model.load("/tf/notebooks/saved_models/cvm-net-2/")


def load_batch(file_names: list,
               sat_dir: str = "/tf/CVUSA/sat_test_cropped",
               gnd_dir: str = "/tf/CVUSA/gnd_test"):

    sat_batch = []
    gnd_batch = []

    for file_name in file_names:
        sat_batch.append(load_and_preprocess_img(os.path.join(sat_dir, file_name), target_size=(224, 224)))
        gnd_batch.append(load_and_preprocess_img(os.path.join(gnd_dir, file_name), target_size=(224, 224)))

    sat_batch = np.array(sat_batch).squeeze()
    gnd_batch = np.array(gnd_batch).squeeze()

    return sat_batch, gnd_batch


if __name__ == "__main__":

    dir_path_gnd = "/tf/CVUSA/gnd_test"

    file_names = os.listdir(dir_path_gnd)
    no_files = len(file_names)
    batch_size = 32

    if not os.path.exists("gnd_descriptors.tsv") or not os.path.exists("sat_descriptors.tsv"):
        gnd_descriptors = []
        sat_descriptors = []

        for i in tqdm(range(0, no_files, batch_size)):
            if i + batch_size >= no_files:
                sat_batch, gnd_batch = load_batch(file_names[i:])
            else:
                sat_batch, gnd_batch = load_batch(file_names[i:i + batch_size])

            gnd_descriptors.append(model.gnd_embedding.predict(gnd_batch, verbose=0))
            sat_descriptors.append(model.sat_embedding.predict(sat_batch, verbose=0))

        # Concatenate batches together into shape (data_length, vector_dimensions)  i.e. (5000, 256)
        gnd_descriptors = np.concatenate(gnd_descriptors, axis=0)
        sat_descriptors = np.concatenate(sat_descriptors, axis=0)

        np.savetxt("gnd_descriptors.tsv", gnd_descriptors, delimiter='\t', fmt='%f', newline='\n')
        np.savetxt("sat_descriptors.tsv", sat_descriptors, delimiter='\t', fmt='%f', newline='\n')

    else:
        gnd_descriptors = np.loadtxt("gnd_descriptors.tsv", delimiter='\t')
        sat_descriptors = np.loadtxt("sat_descriptors.tsv", delimiter='\t')


    print(" Calculating global distance matrix ...", end="")
    global_distances = np.square(cdist(gnd_descriptors, sat_descriptors))
    print(" done")

    results = recall_at_k(global_distances)
    print(f"\nRecall@K:\n"
          f"top   1: {results[0]}\n"
          f"top   5: {results[1]}\n"
          f"top  10: {results[2]}\n"
          f"top  50: {results[3]}\n"
          f"top 100: {results[4]}\n"
          f"top 500: {results[5]}\n"
          f"top  1%: {results[6]}\n"
          f"top  5%: {results[7]}\n"
          f"top 10%: {results[8]}\n")

    for i in range(len(global_distances)):
        min_indx = np.argsort(global_distances[i])[:10]
        if i in min_indx:
            print(min_indx)
            for j in min_indx:
                print(file_names[i], file_names[j])