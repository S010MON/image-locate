import os
from os import listdir

import numpy as np
import tensorflow as tf

from keras.utils import load_img, img_to_array
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.layers import Flatten, Dense, Dropout
from keras import Model, callbacks

from sklearn.model_selection import train_test_split


def init_vgg16():
    # Download the model with weights pre-trained using ImageNet database
    vgg16 = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))

    # Freeze layers for training
    for layer in vgg16.layers:
        layer.trainable = False

    # Create a new 'top' of the model of fully-connected layers
    top_model = vgg16.output
    top_model = Flatten(name="flatten")(top_model)
    top_model = Dense(4096, activation='relu', name="top_dense_1")(top_model)
    top_model = Dropout(0.2)(top_model)
    top_model = Dense(512, activation='relu', name="top_dense_2")(top_model)
    top_model = Dropout(0.2)(top_model)
    top_model = Dense(256, activation='relu', name="top_dense_3")(top_model)
    top_model = Dropout(0.2)(top_model)
    top_model = Dense(128, activation='relu', name="top_dense_4")(top_model)
    top_model = Dropout(0.2)(top_model)
    output_layer = Dense(1, activation='softmax', name="output")(top_model)

    # Group the convolutional base and new fully-connected layers into a Model object.
    model = Model(inputs=vgg16.input, outputs=output_layer)

    # Compiles the model for training.
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def load_and_preprocess_img(path: str):

    img = load_img(path, target_size=(224, 224))
    ary = img_to_array(img)
    ary = np.expand_dims(ary, axis=0)
    ary = preprocess_input(ary)
    return ary[0]


def load_data(path: str, label: int):

    X = []
    y = []

    for filename in listdir(path):
        x = load_and_preprocess_img(path + filename)
        X.append(x)
        y.append(label)

    return X, y


if __name__ == "__main__":

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    FILEPATH_POS = "data/terrestrial/"
    FILEPATH_NEG = "data/classification/0/"

    print("Loading data")
    X, y = load_data(FILEPATH_POS, 1)
    X_neg, y_neg = load_data(FILEPATH_NEG, 0)

    X.extend(X_neg)
    y.extend(y_neg)

    X = np.array(X)
    y = np.array(y)

    print("Split test and train")
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

    # Checkpoint during training
    checkpoint_path = "classifier_chkpts/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                            save_weights_only=True,
                                            verbose=1)

    print("Create model")
    model = init_vgg16()
    model.fit(X_train,
              y_train,
              batch_size=16,
              epochs=10,
              validation_data=(X_test, y_test),
              callbacks=[cp_callback])
