import os
import requests

os.environ[
    'TF_CPP_MIN_LOG_LEVEL'] = '2'  # Bug workaround source: https://stackoverflow.com/questions/38073432/how-to-suppress-verbose-tensorflow-logging
import tensorflow as tf
from tensorflow.python.keras.engine.keras_tensor import KerasTensor

from keras import metrics
from keras import Model, Sequential
from keras.applications import resnet
from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense, Layer, Input, BatchNormalization, Conv2D


def download_file(url, save_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"File downloaded successfully and saved at {save_path}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")


def download_model(model_name: str):
    """
    Download the model from object storage
    """
    valid_models = {"cvm-net", "resnet", "vgg16"}
    if model_name not in valid_models:
        raise ValueError(f"invalid argument 'model_name' was '{model_name}' must be one of the following "
                         f"{str(valid_models)}.")

    files = {
        f"/tf/notebooks/saved_models/{model_name}/fingerprint.pb": f"https://image-locate-models.eu-central-1.linodeobjects.com/{model_name}.fingerprint.pb",
        f"/tf/notebooks/saved_models/{model_name}/keras_metadata.pb": f"https://image-locate-models.eu-central-1.linodeobjects.com/{model_name}.keras_metadata.pb",
        f"/tf/notebooks/saved_models/{model_name}/saved_model.pb": f"https://image-locate-models.eu-central-1.linodeobjects.com/{model_name}.saved_model.pb",
        f"/tf/notebooks/saved_models/{model_name}/variables/variables.index": f"https://image-locate-models.eu-central-1.linodeobjects.com/{model_name}.variables.data-00000-of-00001",
        f"/tf/notebooks/saved_models/{model_name}/variables/variables.data-00000-of-000001": f"https://image-locate-models.eu-central-1.linodeobjects.com/{model_name}.variables.index",
    }

    for path, url in files.items():
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        download_file(url, path)


def embedding(name: str, base: str = 'vgg16', netvlad=False) -> Model:
    """
    Creates an embedding layer that takes an image and maps it to a vector
    space
    :param name: the network name
    :param base: the base network to use (either 'resnet' or 'vgg16')
    :param netvlad: boolean if the NetVLAD module is to be used
    :return:
    """
    if base == 'vgg16':
        base_cnn = VGG16(weights='imagenet',
                         include_top=False,
                         input_shape=(224, 224, 3))

        trainable = False
        for layer in base_cnn.layers:
            if layer.name == "block4_conv1":
                trainable = True
            layer.trainable = trainable

    elif base == 'resnet':
        base_cnn = resnet.ResNet50(weights="imagenet",
                                   input_shape=(200, 200, 3),
                                   include_top=False)

        trainable = False
        for layer in base_cnn.layers:
            if layer.name == "conv5_block2_out":
                trainable = True
            layer.trainable = trainable
    else:
        raise ValueError("Base network must be selected from either 'resnet' or 'vgg16'.")

    if netvlad:
        # Create a NetVLAD top layer
        model = Sequential()
        model.add(base_cnn)
        model.add(NetVLAD(input_shape=base_cnn.output_shape))
        return model

    else:
        # Create a new 'top' of the model of fully-connected layers for Places365
        top = Flatten()(base_cnn.output)
        top = Dense(512, activation="relu", name="fc1")(top)
        top = BatchNormalization()(top)
        top = Dense(256, activation="relu", name="fc2")(top)
        top = BatchNormalization()(top)
        output = Dense(256)(top)
        return Model(base_cnn.input, output, name=name)


class DistanceLayer(Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    source: https://keras.io/examples/vision/siamese_network/
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return ap_distance, an_distance


class NetVLAD(Layer):
    """
    A NetVLAD class for aggregating visual features
    Sources:
        Paper: NetVLAD - CNN architecture for weakly supervised place recognition https://arxiv.org/abs/1511.07247
        Implementation: https://github.com/NHGJem/tensorflow2-NetVLAD/blob/master/netvlad_layer.py
    """

    def __init__(self,
                 input_shape,
                 K=64,
                 assign_weight_initializer=None,
                 cluster_initializer=None,
                 skip_postnorm=False,
                 outdim=32768,
                 **kwargs):
        self.K = K
        self.assign_weight_initializer = assign_weight_initializer
        self.skip_postnorm = skip_postnorm
        self.outdim = outdim
        super(NetVLAD, self).__init__(input_shape=input_shape, **kwargs)

    def build(self, input_shape):
        self.D = input_shape[-1]
        self.C = self.add_weight(name='cluster_centers',
                                 shape=(1, 1, 1, self.D, self.K),
                                 initializer='zeros',
                                 dtype='float32',
                                 trainable=True)

        self.conv = Conv2D(filters=self.K, kernel_size=1, strides=(1, 1),
                           use_bias=True, padding='valid',
                           kernel_initializer='zeros')
        self.conv.build(input_shape)
        self.built = True
        super(NetVLAD, self).build(input_shape)

    def call(self, inputs: KerasTensor):
        if not self.built:
            self.build(input_shape=inputs.shape)
        s = self.conv(inputs)
        a = tf.nn.softmax(s)

        a = tf.expand_dims(a, -2)

        # VLAD core.
        v = tf.expand_dims(inputs, -1) + self.C
        v = a * v
        v = tf.reduce_sum(v, axis=[1, 2])
        v = tf.transpose(v, perm=[0, 2, 1])

        if not self.skip_postnorm:
            # Result seems to be very sensitive to the normalization method
            # details, so sticking to matconvnet-style normalization here.
            v = self.matconvnetNormalize(v, 1e-12)
            v = tf.transpose(v, perm=[0, 2, 1])
            v = self.matconvnetNormalize(Flatten()(v),
                                         1e-12)  # - https://stackoverflow.com/questions/53153790/tensor-object-has-no-attribute-lower

        return v

    def matconvnetNormalize(self, inputs, epsilon):
        return inputs / tf.sqrt(tf.reduce_sum(inputs ** 2, axis=-1, keepdims=True)
                                + epsilon)

    def compute_output_shape(self, input_shape):
        return tuple([None, self.outdim])

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'K': self.K,
            'assign_weight_initializer': self.assign_weight_initializer,
            'skip_postnorm': self.skip_postnorm,
            'outdim': self.outdim
        })
        return config


class SiameseModel(Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, base_network: str = 'resnet', netvlad=False, load_weights=True):
        super().__init__()

        if base_network == 'resnet':
            self.input_dims = (200, 200)
            self.sat_embedding = embedding("Resnet50_Sat_Embedding", base='resnet', netvlad=netvlad)
            self.gnd_embedding = embedding("Resnet50_Gnd_Embedding", base='resnet', netvlad=netvlad)

        elif base_network == 'vgg16':
            self.input_dims = (224, 224)
            self.sat_embedding = embedding("VGG16_Sat_Embedding", base='vgg16', netvlad=netvlad)
            self.gnd_embedding = embedding("VGG16_Gnd_Embedding", base='vgg16', netvlad=netvlad)
        else:
            raise ValueError("Base network not selected.  Please choose from 'resnet' or 'vgg16' base networks")

        anchor_input = Input(name="anchor", shape=self.input_dims + (3,))
        positive_input = Input(name="positive", shape=self.input_dims + (3,))
        negative_input = Input(name="negative", shape=self.input_dims + (3,))

        outputs = [
            self.gnd_embedding(resnet.preprocess_input(anchor_input)),
            self.sat_embedding(resnet.preprocess_input(positive_input)),
            self.sat_embedding(resnet.preprocess_input(negative_input)),
        ]

        self.siamese_network = Model(
            inputs=[anchor_input, positive_input, negative_input], outputs=outputs
        )
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]

    def get_config(self):
        return {self.siamese_network.get_config()}

    def load(self, filepath: str):
        self.siamese_network.load_weights(filepath)
