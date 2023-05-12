import os
import tensorflow as tf
from keras import optimizers
from models import SiameseModel
from utils import load_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print("Tensorflow version:", tf.__version__)
gpus = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))

BATCH_SIZE = 16
train_data = load_data(anchor_images_path="/tf/CVUSA/clean_ground",
                       positive_images_path="/tf/CVUSA/clean_aerial",
                       batch_size=BATCH_SIZE)


WEIGHTS_PATH = "/tf/notebooks/resnet"

# network.load_weights(WEIGHTS_PATH)
model = SiameseModel()
model.compile(optimizer=optimizers.Adam(0.001), weighted_metrics=[])

for epoch in range(1, 11):
    print(f"Epoch: {epoch}")
    model.fit(train_data, epochs=1)
    model.siamese_network.save(WEIGHTS_PATH)