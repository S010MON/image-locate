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

training_path_sat = "/tf/CVUSA/sat_test"
training_path_gnd = "/tf/CVUSA/gnd_test"

input_path_sat = "/tf/CVUSA/sat_train"
input_path_gnd = "/tf/CVUSA/gnd_train"
output_path_sat = "/tf/CVUSA/hier_batch/sat"
output_path_gnd = "/tf/CVUSA/hier_batch/gnd"


def encode_embeddings(file_names: list, input_dir: str, output_file_name: str):
    print("encode embeddings")

    for i, file_name in enumerate(tqdm(file_names)):
        path = os.path.join(input_dir, file_name)
        img_sat = load_and_preprocess_img(path)
        embedding = model.sat_embedding.predict([img_sat], verbose=0)

        with open(output_file_name, "a") as file:
            np.savetxt(file, embedding, delimiter='\t', fmt='%f', newline='\n')


def k_means_model(n_clusters: int, embeddings: np.ndarray) -> (np.ndarray, MiniBatchKMeans):
    # Fit K-Means clusters
    clf = MiniBatchKMeans(n_clusters=n_clusters,
                          random_state=0,
                          batch_size=64,
                          n_init='auto')
    clf.fit(embeddings)
    return clf


def move_files(input_dir_sat: str, input_dir_gnd, file_names: list, preds: np.ndarray, n_clusters: int):

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
        src_sat = os.path.join(input_dir_sat, file_name)
        dest_sat = os.path.join(output_path_sat, str(preds[i]), file_name)
        shutil.copy2(src_sat, dest_sat)

        src_gnd = os.path.join(input_dir_gnd, file_name)
        dest_gnd = os.path.join(output_path_gnd, str(preds[i]), file_name)
        shutil.copy2(src_gnd, dest_gnd)


if __name__ == "__main__":
    no_clusters = 3
    training_file_names = os.listdir(training_path_sat)

    if not os.path.exists("embeddings_subset.tsv") or not os.path.isfile("embeddings_subset.tsv"):
        encode_embeddings(training_file_names,
                          input_dir=training_path_sat,
                          output_file_name="embeddings_subset.tsv")    # This writes to file due to size

    # Train the classifier
    embeddings = np.loadtxt("embeddings_subset.tsv", delimiter='\t')
    print("K-means clustering ... ", end="")
    clf = k_means_model(no_clusters, embeddings)
    predictions = clf.predict(embeddings)

    if not os.path.exists("predictions_subset.tsv") or not os.path.isfile("predictions_subset.tsv"):
        with open("predictions_subset.tsv", "w") as file:
            file.write("file_name\tprediction\n")
            for file_name, prediction in zip(training_file_names, predictions):
                file.write(f"{file_name}\t{prediction}\n")

    move_files(training_path_sat, training_path_gnd, training_file_names, predictions, no_clusters)
    print(" Complete")

    exit(0)

    file_names = os.listdir(input_path_sat)
    encode_embeddings(file_names,
                      input_dir=input_path_sat,
                      output_file_name="embeddings.tsv")

    # Predict the class on whole dataset
    with open("embeddings.tsv", "r") as embeddings_file:
        count = 0
        line = embeddings_file.readline()
        while line and line != "":
            print(f"\rpredicting: {count}", end="")
            array = np.fromstring(line, sep='\t', dtype=float).reshape(1, -1)
            prediction = clf.predict(array)
            with open("predictions.tsv", "a") as predictions_file:
                predictions_file.write(f"{prediction}\n")
            line = embeddings_file.readline()
            count += 1

    predictions = np.loadtxt('predictions.tsv', delimiter='\t')
    move_files(file_names, predictions, no_clusters)
