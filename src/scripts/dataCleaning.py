import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from keras import Model
from keras.layers import Flatten, Dense, Dropout
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.utils import load_img, img_to_array, get_file, image_dataset_from_directory

from sklearn.ensemble import RandomForestClassifier
from cvusa import get_metadata


def init_vgg16():
    vgg16 = VGG16(weights=None,
                  include_top=False,
                  input_shape=(224, 224, 3))

    # Create a new 'top' of the model of fully-connected layers for Places365
    top_model = vgg16.output
    top_model = Flatten(name="flatten")(top_model)
    top_model = Dense(4096, activation='relu', name="fc1")(top_model)
    top_model = Dropout(0.5, name="drop_fc1")(top_model)
    top_model = Dense(4096, activation='relu', name="fc2")(top_model)
    top_model = Dropout(0.2, name="drop_fc2")(top_model)
    output_layer = Dense(365, activation='softmax', name="predictions")(top_model)

    model = Model(inputs=vgg16.input,
                  outputs=output_layer,
                  name="vgg16-places365")

    WEIGHTS_PATH = 'https://github.com/GKalliatakis/Keras-VGG16-places365/releases/download/v1.0/vgg16' \
                   '-places365_weights_tf_dim_ordering_tf_kernels.h5'

    weights_path = get_file('vgg16-places365_weights_tf_dim_ordering_tf_kernels.h5',
                            WEIGHTS_PATH,
                            cache_subdir='models')

    model.load_weights(weights_path)

    return model


def preprocess_labelled_image(image: tf.Tensor, label) -> tuple:
    return tf.divide(image, 255), label


def load_and_preprocess_img(path: str):

    img = load_img(path, target_size=(224, 224))
    ary = img_to_array(img)
    ary = np.expand_dims(ary, axis=0)
    ary = preprocess_input(ary)
    return ary


def load_data(filepath: str) -> tf.data.Dataset:
    dataset = image_dataset_from_directory(filepath,
                                           color_mode='rgb',
                                           image_size=(224, 224))
    return dataset


def train_random_forest(x_path: str, y_path: str):
    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path).squeeze()
    assert len(X) == len(y)

    clf = RandomForestClassifier(n_estimators=300,
                                 random_state=42,
                                 n_jobs=-1)
    clf.fit(X.values, y.values)
    return clf


if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    print("Loading Model")
    model = init_vgg16()

    preds_file_path = "/tf/notebooks/classification_data.csv"
    label_file_path = "/tf/notebooks/classification_labels.csv"

    if not os.path.exists(preds_file_path) or not os.path.exists(label_file_path):
        print("Loading Classifier Training Data")
        data = load_data("/tf/notebooks/data/classification")

        for images, labels in data:
            preds = model.predict(images)
            with open(preds_file_path, 'ab') as file:
                np.savetxt(file, preds, delimiter=',')

            with open(label_file_path, 'ab') as file:
                int_labels = labels.numpy().astype("uint8")
                np.savetxt(file, int_labels, delimiter=',', fmt='%i')

    print("Create Random Forest")
    clf = train_random_forest(preds_file_path, label_file_path)

    with open('/tf/CVUSA/flickr_images.txt', 'r') as f:
        flickr_data = [(x.strip(),) + get_metadata(x.strip()) for x in f]

    # data in tuple form
    # ('flickr/39/-100/37603091@N02_3832037528_39.353314_-100.441957.jpg', '39.353314', '-100.441957', '37603091@N02',
    #   '3832037528', 'https://www.flickr.com/photos/37603091@N02/3832037528')

    # Run through data and classify images
    print(f"Data: {len(flickr_data)} files")
    count = 0

    for data in tqdm(flickr_data):
        count += 1

        filepath = "/tf/CVUSA/" + data[0]
        filename = filepath.split('/')[-1]
        print(f"{data[0]} - {count}/{len(flickr_data)}: {round(100* (count/len(flickr_data)), 0)}%")

        img = load_and_preprocess_img(filepath)
        features = model.predict(img, verbose=0)
        features = features[0].reshape(1, -1)
        classification = clf.predict(features)

        if classification:
            with open("/tf/CVUSA/flickr_clean.txt", "a") as file:
                file.write(f"{data[0]}\n")


