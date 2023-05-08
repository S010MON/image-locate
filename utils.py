import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt




def preprocess_image(image: tf.Tensor) -> tf.Tensor:
    image = tf.divide(image, 255.0)
    return image


def preprocess_triplets(anchor: tf.Tensor, positive: tf.Tensor, negative: tf.Tensor) -> tuple:
    """
    Given three filepaths, load and preprocess each and return the tuple of
    all three.
    :param anchor: the anchor image path
    :param positive: the positive image path
    :param negative: the negative image path
    :return: tuple of three preprocessed images
    """

    return (preprocess_image(anchor),
            preprocess_image(positive),
            preprocess_image(negative))


def load_data(batch_size: int = 16) -> tf.data.Dataset:
    anchor_images_path = "/tf/CVUSA/terrestrial/"
    positive_images_path = "/tf/CVUSA/satellite_polar/"

    # TODO This will need to be improved using distance measures

    # Create datasets
    anchor_dataset = tf.keras.utils.image_dataset_from_directory(anchor_images_path,
                                                                 label_mode=None,
                                                                 color_mode='rgb',
                                                                 image_size=(224, 224),
                                                                 batch_size=batch_size,
                                                                 shuffle=False)

    positive_dataset = tf.keras.utils.image_dataset_from_directory(positive_images_path,
                                                                   label_mode=None,
                                                                   color_mode='rgb',
                                                                   image_size=(224, 224),
                                                                   batch_size=batch_size,
                                                                   shuffle=False)

    negative_dataset = tf.keras.utils.image_dataset_from_directory(positive_images_path,
                                                                   label_mode=None,
                                                                   color_mode='rgb',
                                                                   image_size=(224, 224),
                                                                   batch_size=batch_size,
                                                                   shuffle=True,
                                                                   seed=42)

    dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
    dataset = dataset.shuffle(buffer_size=32)
    dataset = dataset.map(preprocess_triplets)
    dataset = dataset.prefetch(8)

    return dataset


def visualise(anchor, positive, negative):
    """Visualise a few triplets from the supplied batches."""

    def show(ax, image):
        ax.imshow(image)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    fig = plt.figure(figsize=(9, 9))

    axs = fig.subplots(3, 3)
    for i in range(3):
        show(axs[i, 0], anchor[i])
        show(axs[i, 1], positive[i])
        show(axs[i, 2], negative[i])
