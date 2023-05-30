import os, sys, shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

os.environ[
    'TF_CPP_MIN_LOG_LEVEL'] = '2'  # Bug workaround source: https://stackoverflow.com/questions/38073432/how-to-suppress-verbose-tensorflow-logging
import tensorflow as tf
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

from models import SiameseModel
from dataCleaning import load_and_preprocess_img

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

model = SiameseModel(base_network='vgg16', netvlad=True)

input_path_sat = "/tf/CVUSA/satellite"
input_path_gnd = "/tf/CVUSA/terrestrial"
output_path_sat = "/tf/CVUSA/hier_batch/sat"
output_path_gnd = "/tf/CVUSA/hier_batch/gnd"

file_names = os.listdir(input_path_sat)
embeddings = np.zeros((len(file_names), 32768))

print("Encode embeddings")

# Encode embeddings for all images in the folder
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

# Write to file just in case / for visualisation
print("Write to file")
with open("hierarchical_batches.csv", "w") as file:
    file.write("file_name,prediction,dims")
    for i, file_name in enumerate(tqdm(file_names)):
        embedding_str = ",".join(map(str, embeddings[i, :]))
        file.write(f"{file_name},{preds[i]},{embedding_str}\n")

# Make sure destination files exist
for n in range(n_clusters):
    if not os.path.exists(os.path.join(output_path_sat, str(n))):
        os.mkdir(os.path.join(output_path_sat, str(n)))

    if not os.path.exists(os.path.join(output_path_gnd, str(n))):
        os.mkdir(os.path.join(output_path_gnd, str(n)))

# Copy files over to new homes
for i, file_name in enumerate(tqdm(file_names)):
    src_sat = os.path.join(input_path_sat, file_name)
    dest_sat = os.path.join(output_path_sat, file_name)
    shutil.copy2(src_sat, dest_sat)

    src_gnd = os.path.join(input_path_gnd, file_name)
    dest_gnd = os.path.join(output_path_gnd, file_name)
    shutil.copy2(src_sat, dest_sat)
