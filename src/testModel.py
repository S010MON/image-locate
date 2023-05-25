import numpy as np
from keras import optimizers
from tqdm import tqdm

from models import SiameseModel
from utils import load_data, distance_matrix, recall_at_k

# --- Set global variables --- #
BATCH_SIZE = 16
EPOCHS = 10
WEIGHTS_PATH = "/tf/notebooks/resnet"


def test(model=None):
    test_data = load_data(anchor_images_path="/tf/CVUSA/terrestrial",
                          positive_images_path="/tf/CVUSA/satellite",
                          batch_size=BATCH_SIZE)

    # Allows for loading a model in for testing at the end of each epoch
    if model is None:
        model = SiameseModel()
        model.load(WEIGHTS_PATH)
        optimiser = optimizers.Adam(0.001)
        model.compile(optimizer=optimiser, weighted_metrics=[])

    # Separate the twins for testing
    gnd_embedding = model.gnd_embedding
    sat_embedding = model.sat_embedding

    # Global descriptors will hold each batch of embeddings temporarily to allow batched predictions
    global_sat_descriptors = []             # [ [e_1, e_2, ... e_n], [e_1, e_2 ... e_n] ]
    global_gnd_descriptors = []             # where e = embedding vector and n = batch_size

    for gnd, sat_p, sat_n in tqdm(test_data.as_numpy_iterator()):
        global_gnd_descriptors.append(gnd_embedding.predict(gnd, verbose=0))
        global_sat_descriptors.append(sat_embedding.predict(sat_p, verbose=0))

    # Concatenate batches together into shape (data_length, vector_dimensions)  i.e. (5000, 256)
    global_gnd_descriptors = np.concatenate(global_gnd_descriptors)
    global_sat_descriptors = np.concatenate(global_sat_descriptors)

    global_distances = distance_matrix(global_gnd_descriptors, global_sat_descriptors)

    results = recall_at_k(global_distances)
    print(results)



if __name__ == "__main__":
    test()
