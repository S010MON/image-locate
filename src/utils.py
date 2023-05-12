import math
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


def load_data(anchor_images_path: str="/tf/CVUSA/clean_ground/",
              positive_images_path: str="/tf/CVUSA/clean_aerial/",
              input_shape=(200, 200),
              batch_size: int = 16) -> tf.data.Dataset:

    # Create datasets
    anchor_dataset = tf.keras.utils.image_dataset_from_directory(anchor_images_path,
                                                                 label_mode=None,
                                                                 color_mode='rgb',
                                                                 image_size=input_shape,
                                                                 batch_size=batch_size,
                                                                 shuffle=False)

    positive_dataset = tf.keras.utils.image_dataset_from_directory(positive_images_path,
                                                                   label_mode=None,
                                                                   color_mode='rgb',
                                                                   image_size=input_shape,
                                                                   batch_size=batch_size,
                                                                   shuffle=False)

    negative_dataset = tf.keras.utils.image_dataset_from_directory(positive_images_path,
                                                                   label_mode=None,
                                                                   color_mode='rgb',
                                                                   image_size=input_shape,
                                                                   batch_size=batch_size,
                                                                   shuffle=True,
                                                                   seed=42)

    dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
    dataset = dataset.shuffle(buffer_size=64)
    dataset = dataset.map(preprocess_triplets)
    dataset = dataset.prefetch(16)
    return dataset


def sample_within_bounds(signal: np.ndarray, x, y, bounds):
    """
    Source: Where am I looking at? Joint Location and Orientation Estimation by Cross-View Matching CVPR2020
    Yujiao Shi, Xin Yu, Dylan Campbell, Hongdong Li.
    https://github.com/shiyujiao/cross_view_localization_DSM/blob/master/script/data_preparation.py
    """
    xmin, xmax, ymin, ymax = bounds

    idxs = (xmin <= x) & (x < xmax) & (ymin <= y) & (y < ymax)

    sample = np.zeros((x.shape[0], x.shape[1], signal.shape[-1]))
    sample[idxs, :] = signal[x[idxs], y[idxs], :]

    return sample


def sample_bilinear(signal: np.ndarray, rx, ry):
    """
    Source: Where am I looking at? Joint Location and Orientation Estimation by Cross-View Matching CVPR2020
    Yujiao Shi, Xin Yu, Dylan Campbell, Hongdong Li.
    https://github.com/shiyujiao/cross_view_localization_DSM/blob/master/script/data_preparation.py
    """

    signal_dim_x = signal.shape[0]
    signal_dim_y = signal.shape[1]

    # obtain four sample coordinates
    ix0 = rx.astype(int)
    iy0 = ry.astype(int)
    ix1 = ix0 + 1
    iy1 = iy0 + 1

    bounds = (0, signal_dim_x, 0, signal_dim_y)

    # sample signal at each four positions
    signal_00 = sample_within_bounds(signal, ix0, iy0, bounds)
    signal_10 = sample_within_bounds(signal, ix1, iy0, bounds)
    signal_01 = sample_within_bounds(signal, ix0, iy1, bounds)
    signal_11 = sample_within_bounds(signal, ix1, iy1, bounds)

    na = np.newaxis
    # linear interpolation in x-direction
    fx1 = (ix1 - rx)[..., na] * signal_00 + (rx - ix0)[..., na] * signal_10
    fx2 = (ix1 - rx)[..., na] * signal_01 + (rx - ix0)[..., na] * signal_11

    # linear interpolation in y-direction
    return (iy1 - ry)[..., na] * fx1 + (ry - iy0)[..., na] * fx2


def polar(img, output_shape=(512,512)):
    S = img.shape[0]  # Original size of the aerial image
    height = output_shape[0]  # Height of polar transformed aerial image
    width = output_shape[1]  # Width of polar transformed aerial image

    i = np.arange(0, height)
    j = np.arange(0, width)
    jj, ii = np.meshgrid(j, i)

    y = S / 2. - S / 2. / height * (height - 1 - ii) * np.sin(2 * np.pi * jj / width)
    x = S / 2. + S / 2. / height * (height - 1 - ii) * np.cos(2 * np.pi * jj / width)

    return sample_bilinear(img, x, y) / 255


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


def display(images, axis='off', cmap=None):

    fig = plt.figure(figsize=(15,10))
    cols = 2
    rows = math.ceil(len(images)/2)

    for i in range(len(images)):
        fig.add_subplot(rows, cols, i+1)
        plt.imshow(images[i], cmap=cmap)
        plt.axis(axis)