import os
import numpy as np
import tensorflow as tf
from keras import optimizers
from models import SiameseModel
from utils import load_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print("Tensorflow version:", tf.__version__)
gpus = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))

BATCH_SIZE = 16
train_data = load_data(anchor_images_path="/tf/CVUSA/terrestrial",
                       positive_images_path="/tf/CVUSA/satellite",
                       batch_size=BATCH_SIZE)


WEIGHTS_PATH = "/tf/notebooks/resnet"

# network.load_weights(WEIGHTS_PATH)
model = SiameseModel()
optimiser = optimizers.Adam(0.001)
model.compile(optimizer=optimiser, weighted_metrics=[])

for epoch in range(10):
    print(f"\nEpoch: {epoch} {train_data.__len__()}")

    total_loss = -1

    for step, (a, p, n) in enumerate(train_data.as_numpy_iterator()):

        if a.shape != p.shape != n.shape:
            continue
        # Mine hard triplets
        n = model.mine_hard_triplets(a, p, n)

        with tf.GradientTape() as tape:
            # Forward pass on the Hard Triplets
            ap_distance, an_distance = model.siamese_network((a, p, n))

            # Compute the loss
            loss = ap_distance - an_distance
            loss = tf.maximum(loss + model.margin, 0.0)
            if total_loss == -1:
                total_loss = np.mean(loss)
            else:
                total_loss += np.mean(loss)

        grads = tape.gradient(loss, model.siamese_network.trainable_weights)
        optimiser.apply_gradients(zip(grads, model.siamese_network.trainable_weights))

        print(f"\rbatch: {step}/{train_data.__len__()} loss: {np.round(total_loss / step)}", end="")

    print(f"\nsaving weights to: {WEIGHTS_PATH}")
    model.siamese_network.save(WEIGHTS_PATH)