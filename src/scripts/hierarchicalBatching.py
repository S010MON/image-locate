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

model = SiameseModel(base_network='resnet', netvlad=False)
model.load("/tf/notebooks/saved_models/resnet/")

input_path_sat = "/tf/CVUSA/sat_train"
input_path_gnd = "/tf/CVUSA/gnd_train"
output_path_sat = "/tf/CVUSA/hier_batch/sat"
output_path_gnd = "/tf/CVUSA/hier_batch/gnd"


def encode_embeddings(file_names: list):
    print("encode embeddings")

    for i, file_name in enumerate(tqdm(file_names)):
        path = os.path.join(input_path_sat, file_name)
        img_sat = load_and_preprocess_img(path, target_size=(200, 200))
        embedding = model.sat_embedding.predict([img_sat], verbose=0)

        with open("embeddings.tsv", "a") as file:
            np.savetxt(file, embedding, delimiter='\t', fmt='%f', newline='\n')


def k_means_model(n_clusters: int, embeddings: np.ndarray) -> (np.ndarray, MiniBatchKMeans):
    # Fit K-Means clusters
    clf = MiniBatchKMeans(n_clusters=n_clusters,
                          random_state=0,
                          batch_size=64,
                          n_init='auto')
    clf.fit(embeddings)
    return clf


def move_files(file_names: list, preds: np.ndarray, n_clusters: int):

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
    file_names = os.listdir(input_path_sat)
    encode_embeddings(file_names)

    # Take a subset of embeddings to train the classifier
    with open("embeddings.tsv", "r") as main_file:
        count = 0
        limit = 50000
        while count <= limit:
            print(f"\rsubset: {count}/{limit}", end="")
            with open("embeddings_subset.tsv", "a") as subset_file:
                subset_file.write(main_file.readline())
                subset_file.write("\n")
            count += 1
    print("")

    # Train the classifer
    embeddings = np.loadtxt("embeddings_subset.tsv", delimiter='\t')
    clf = k_means_model(3, embeddings)

    # Predict the class on whole dataset
    with open("embeddings.tsv", "r") as embeddings_file:

        count = 0
        line = embeddings_file.readline()
        while line:
            print(f"\rpredicting: {count}", end="")
            array = np.fromstring(line, sep='\t', dtype=float).reshape(1, -1)
            prediction = clf.predict(array)
            with open("predictions.tsv", "a") as predictions_file:
                predictions_file.write(f"{prediction}\n")
            line = embeddings_file.readline()
            count += 1

    # for idx in range(0, len(file_names), chunk_size):
    #     print(f"Processing chunk {idx // chunk_size}")
    #     end_idx = idx + chunk_size
    #
    #     if end_idx >= len(file_names):
    #         end_idx = len(file_names) - 1
    #
    #     super_batch(file_names[idx: end_idx])
    #
    # print("complete")
    # exit(0)