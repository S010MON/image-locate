import os
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
from utils import load_and_preprocess_img



def encode_embeddings(file_names: list, input_dir: str, output_file_name: str):
    print("Encoding embeddings")

    for i, file_name in enumerate(tqdm(file_names)):
        path = os.path.join(input_dir, file_name)
        img_sat = load_and_preprocess_img(path)
        embedding = model.sat_embedding.predict([img_sat], verbose=0)

        with open(output_file_name, "a") as file:
            np.savetxt(file, embedding, delimiter='\t', fmt='%f', newline='\n')


def build_tree(node_path: str, k: int = 3, leaf_size: int = 100, subset_size: int = 50000):
    """
    :param node_path: the path to the current node in the tree
    :param k: the number of clusters per node (default 3)
    :param leaf_size: the number of images in a node where the tree will no longer split
    """
    input_file = os.path.join(node_path, "input.tsv")
    embeddings_file = os.path.join(node_path, "embeddings.tsv")

    with open(input_file, "r") as file:
        count = sum(1 for _ in file)

    print(f"{node_path}\t\t {count} embeddings")
    if count < leaf_size:
        return

    # If we have a large dataset - take a subset of `subset_size` values
    if count > subset_size:
        print("Create subset")
        with open(embeddings_file, "r") as read_file:
            with open(os.path.join(node_path, "embeddings_subset.tsv"), "a") as write_file:
                i = 0
                line = read_file.read()
                while line and i < subset_size:
                    write_file.write(line)
                    i += 1
        embeddings_trg = np.loadtxt(os.path.join(node_path, "embeddings_subset.tsv"), delimiter='\t')
    else:
        embeddings_trg = np.loadtxt(os.path.join(node_path, "embeddings.tsv"), delimiter='\t')

    # Train classifier
    clf = MiniBatchKMeans(n_clusters=k, random_state=0, batch_size=64, n_init='auto')
    clf.fit(embeddings_trg)

    # Write cluster centroids to file
    means = clf.cluster_centers_
    with open(os.path.join(node_path, "means.tsv"), "w") as mean_file:
        np.savetxt(mean_file, means, delimiter='\t', fmt='%f', newline='\n')

    # Make sure destination dirs exist
    for i in range(k):
        if not os.path.exists(os.path.join(node_path, str(i))):
            os.makedirs(os.path.join(node_path, str(i)))

    # Write predictions to sub-trees
    with open(input_file, "r") as input_file:
        with open(embeddings_file, "r") as file:
            file_name = input_file.readline()
            embedding = file.readline()
            while embedding and embedding != "":
                array = np.fromstring(embedding, sep='\t', dtype=float).reshape(1, -1)
                prediction = clf.predict(array)
                sub_path = os.path.join(node_path, str(prediction[0]))
                with open(os.path.join(sub_path, "input.tsv"), "a") as sub_file:
                    sub_file.write(f"{file_name}")
                with open(os.path.join(sub_path, "embeddings.tsv"), "a") as sub_file:
                    sub_file.write(f"{embedding}")

                embedding = file.readline()
                file_name = input_file.readline()

    for i in range(k):
        sub_path = os.path.join(node_path, str(i))
        build_tree(sub_path)


def recall(node_path: str, query: np.ndarray):
    means = np.loadtxt(os.path.join(node_path, "means.tsv"), delimiter='\t')
    distances = np.square(np.norm(np.subtract(means, query)))
    min_idx = np.argmin(distances)
    return min_idx


if __name__ == "__main__":

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    sat_dir = "/tf/CVUSA/sat_test/"
    file_names = os.listdir(sat_dir)

    # if there is no input list -> create the list
    if not os.path.exists("input.tsv"):
        with open("input.tsv", "w") as file:
            for file_name in os.listdir(sat_dir):
                file.write(f"{file_name}\n")

    model = SiameseModel(base_network='vgg16', netvlad=True)

    # If there are no embeddings -> create an embeddings file from the list
    if not os.path.exists("embeddings.tsv") or not os.path.isfile("embeddings.tsv"):
        model.load("/tf/notebooks/saved_models/resnet/")
        encode_embeddings(file_names, input_dir=sat_dir, output_file_name="embeddings.tsv")

    # Recursively build the tree
    root_node = "/tf/notebooks/src/tree"
    build_tree(root_node)

    # Test against all ground images
    gnd_dir = "/tf/CVUSA/gnd_test/"
    file_names = os.listdir(gnd_dir)

    for file_name in file_names:
        path = os.path.join(gnd_dir, file_name)
        image = load_and_preprocess_img(path)
        embedding = model.gnd_embedding([image], verbose=0)
        result = recall(root_node, embedding)
        print(result)
