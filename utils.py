import tensorflow as tf


def preprocess_image(image: tf.Tensor) -> tf.Tensor:
    image = tf.divide(image, 255)
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


def load_data(batch_size: int = 32) -> tf.data.Dataset:
    anchor_images_path = "/tf/notebooks/data/terrestrial/"
    positive_images_path = "/tf/notebooks/data/satellite/"

    # TODO This will need to be improved using distance measures

    # Create datasets
    anchor_dataset = tf.keras.utils.image_dataset_from_directory(anchor_images_path,
                                                                 label_mode=None,
                                                                 color_mode='rgb',
                                                                 image_size=(224, 224),
                                                                 shuffle=False)

    positive_dataset = tf.keras.utils.image_dataset_from_directory(positive_images_path,
                                                                   label_mode=None,
                                                                   color_mode='rgb',
                                                                   image_size=(224, 224),
                                                                   shuffle=False)

    negative_dataset = tf.keras.utils.image_dataset_from_directory(positive_images_path,
                                                                   label_mode=None,
                                                                   color_mode='rgb',
                                                                   image_size=(224, 224),
                                                                   shuffle=True,
                                                                   seed=42)

    dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
    # dataset = dataset.shuffle(buffer_size=516)
    dataset = dataset.map(preprocess_triplets)
    # dataset = dataset.batch(batch_size=batch_size, drop_remainder=False)
    dataset = dataset.prefetch(8)

    return dataset
