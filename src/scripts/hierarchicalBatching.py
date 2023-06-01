import os
import shutil
import sys
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans

os.environ[
    'TF_CPP_MIN_LOG_LEVEL'] = '2'  # Bug workaround source: https://stackoverflow.com/questions/38073432/how-to-suppress-verbose-tensorflow-logging
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # Allows the importing of parent modules
from models import SiameseModel
from dataCleaning import load_and_preprocess_img

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

model = SiameseModel(base_network='vgg16', netvlad=True)
model.load("/tf/notebooks/saved_models/cvm-net/")

input_path_sat = "/tf/CVUSA/sat_train"
input_path_gnd = "/tf/CVUSA/gnd_train"
output_path_sat = "/tf/CVUSA/hier_batch/sat"
output_path_gnd = "/tf/CVUSA/hier_batch/gnd"


def super_batch(file_names: list):
    # Encode embeddings for all images in the list
    embeddings = np.zeros((len(file_names), 32768))
    for i, file_name in enumerate(tqdm(file_names)):
        path = os.path.join(input_path_sat, file_name)
        img_sat = load_and_preprocess_img(path)
        embeddings[i, :] = model.sat_embedding.predict([img_sat], verbose=0)

    # Fit K-Means clusters
    n_clusters = 3
    kmeans = MiniBatchKMeans(n_clusters=n_clusters,
                             random_state=0,
                             batch_size=64,
                             n_init='auto')
    preds = kmeans.fit_predict(embeddings)

    # # Write to file just in case / for visualisation
    # with open("hierarchical_batches.csv", "a") as file:
    #     file.write("file_name,prediction,dims")
    #     for i, file_name in enumerate(tqdm(file_names)):
    #         embedding_str = ",".join(map(str, embeddings[i, :]))
    #         file.write(f"{file_name},{preds[i]},{embedding_str}\n")

    # Make sure destination files exist
    for n in range(n_clusters):
        if not os.path.exists(os.path.join(output_path_sat, str(n))):
            sat_dir_n = os.path.join(output_path_sat, str(n))
            print("Making dir: ", sat_dir_n)
            os.makedirs(sat_dir_n)

        if not os.path.exists(os.path.join(output_path_gnd, str(n))):
            gnd_dir_n = os.path.join(output_path_gnd, str(n))
            print("Making dir: ", gnd_dir_n)
            os.makedirs(gnd_dir_n)

    # Copy files over to new homes
    for i, file_name in enumerate(tqdm(file_names)):
        src_sat = os.path.join(input_path_sat, file_name)
        dest_sat = os.path.join(output_path_sat, str(preds[i]), file_name)
        shutil.copy2(src_sat, dest_sat)

        src_gnd = os.path.join(input_path_gnd, file_name)
        dest_gnd = os.path.join(output_path_gnd, str(preds[i]), file_name)
        shutil.copy2(src_gnd, dest_gnd)


if __name__ == "__main__":
    chunk_size = 10000
    file_names = os.listdir(input_path_sat)
    for idx in range(0, len(file_names), chunk_size):
        print(f"Processing chunk {idx // chunk_size}")
        end_idx = idx + chunk_size

        if end_idx >= len(file_names):
            end_idx = len(file_names) - 1

        super_batch(file_names[idx: end_idx])

    print("complete")
    exit(0)