import tensorflow as tf


class Dataset:

    def __init__(self,
                 gnd_images_path: str = "/tf/CVUSA/clean_ground/",
                 sat_images_path: str = "/tf/CVUSA/clean_aerial/",
                 base_network: str = "vgg16",
                 batch_size: int = 16,
                 random_crop: bool = True):

        self.gnd_images_path = gnd_images_path
        self.sat_images_path = sat_images_path
        self.base_network = base_network
        self.batch_size = batch_size
        self.random_crop = random_crop
        self.pre_crop_shape = (300, 300)

        if base_network == "vgg16":
            self.input_shape = (224, 224)
        elif base_network == "resnet":
            self.input_shape = (200, 200)
        else:
            raise ValueError(f"Incorrect argument provided for 'base_network' found {base_network}, must be selected "
                             f"from 'vgg16' or 'resnet'")

    def _preprocess_image(self, image: tf.Tensor) -> tf.Tensor:
        image = tf.divide(image, 255.0)
        size = (self.batch_size, self.input_shape[0], self.input_shape[1], 3)
        image = tf.image.random_crop(image, size=size)
        return image

    def _preprocess_triplets(self, gnd: tf.Tensor, sat_pos: tf.Tensor, sat_neg: tf.Tensor) -> tuple:
        """
        Given three filepaths, load and preprocess each and return the tuple of
        all three.
        :param gnd: the anchor image path
        :param sat_pos: the positive image path
        :param sat_neg: the negative image path
        :return: tuple of three preprocessed images
        """
        return (self._preprocess_image(gnd),
                self._preprocess_image(sat_pos),
                self._preprocess_image(sat_neg))

    def load(self) -> tf.data.Dataset:

        if not self.random_crop:
            pre_crop_shape = self.input_shape
        else:
            pre_crop_shape = self.pre_crop_shape

        anchor_dataset = tf.keras.utils.image_dataset_from_directory(self.gnd_images_path,
                                                                     label_mode=None,
                                                                     color_mode='rgb',
                                                                     image_size=self.input_shape,
                                                                     batch_size=self.batch_size,
                                                                     shuffle=False)

        positive_dataset = tf.keras.utils.image_dataset_from_directory(self.sat_images_path,
                                                                       label_mode=None,
                                                                       color_mode='rgb',
                                                                       image_size=pre_crop_shape,
                                                                       batch_size=self.batch_size,
                                                                       shuffle=False)

        negative_dataset = tf.keras.utils.image_dataset_from_directory(self.sat_images_path,
                                                                       label_mode=None,
                                                                       color_mode='rgb',
                                                                       image_size=pre_crop_shape,
                                                                       batch_size=self.batch_size,
                                                                       shuffle=True,
                                                                       seed=42)

        dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
        dataset = dataset.shuffle(buffer_size=64)
        dataset = dataset.map(self._preprocess_triplets)
        dataset = dataset.prefetch(16)
        return dataset
