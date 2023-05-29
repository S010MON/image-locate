import numpy as np
import scipy.spatial.distance
import tensorflow as tf
from tensorflow.python.keras.engine.keras_tensor import KerasTensor

from keras import metrics
from keras import Model, Sequential
from keras.applications import resnet
from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense, Layer, Input, BatchNormalization, Conv2D


def embedding(name: str, base: str = 'resnet', netvlad=False) -> Model:
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

        for layer in base_cnn.layers:
            layer.trainable = False

    elif base == 'resnet':
        base_cnn = resnet.ResNet50(weights="imagenet",
                                   input_shape=(200, 200, 3),
                                   include_top=False)

        trainable = False
        for layer in base_cnn.layers:
            if layer.name == "conv5_block1_out":
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


def resnet_embedding(name: str, netvlad=True) -> Model:
    base_cnn = resnet.ResNet50(
        weights="imagenet",
        input_shape=(200, 200, 3),
        include_top=False
    )
    trainable = False
    for layer in base_cnn.layers:
        if layer.name == "conv5_block1_out":
            trainable = True
        layer.trainable = trainable

    if netvlad:
        model = Sequential()
        model.add(base_cnn)
        model.add(NetVLAD(input_shape=base_cnn.output_shape))
        return model

    else:
        flatten = Flatten()(base_cnn.output)
        dense1 = Dense(512, activation="relu")(flatten)
        dense1 = BatchNormalization()(dense1)
        dense2 = Dense(256, activation="relu")(dense1)
        dense2 = BatchNormalization()(dense2)
        output = Dense(256)(dense2)
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

    def __init__(self, base_network: str = 'resnet', netvlad=False):
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

        distances = DistanceLayer()(
            self.gnd_embedding(resnet.preprocess_input(anchor_input)),
            self.sat_embedding(resnet.preprocess_input(positive_input)),
            self.sat_embedding(resnet.preprocess_input(negative_input)),
        )

        self.siamese_network = Model(
            inputs=[anchor_input, positive_input, negative_input], outputs=distances
        )
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def mine_hard_triplets(self, anchor: np.ndarray, positive: np.ndarray, negative: np.ndarray) -> tuple:
        """
        Online mine semi-hard triplets where d(a,p) < d(a,n) < d(a,p) + margin
        :param anchor: a (batch_size, width, height, 3) tensor of Anchor images
        :param positive: a (batch_size, width, height, 3) tensor of Positive images
        :param negative: a (batch_size, width, height, 3) tensor of Negative images
        :return: A reordered tuple of (A, P, N_hard) where A and P are unchanged and
                N_hard are semi hard images
        """
        # Calculate embeddings for each of the images in the batch
        anchor_embedding = self.gnd_embedding(anchor)
        positive_embedding = self.sat_embedding(positive)
        negative_embedding = self.sat_embedding(negative)

        # Calculate the distance between the anchor and P/A embeddings
        dp = anchor_embedding - positive_embedding
        dn = anchor_embedding - negative_embedding

        # Create a distance matrix between every element of the batch ||dp.T - dn||^2
        # shape: (batch_size, batch_size)
        distance = np.square(scipy.spatial.distance.cdist(dp, dn))

        # Create a mask to remove all the diagonal of the matrix (i.e. where i == j)
        mask = tf.ones(tf.shape(distance)) - tf.eye(tf.shape(distance)[0])
        distance = tf.multiply(distance, mask)

        # Take the argmax of each row to find the hardest image to complete the triplet
        indx = tf.argmax(distance, axis=1)
        return tf.gather(negative, indx)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.

        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        print("data shape ", data)
        anchor_embeddings = self.gnd_embedding(data[0])
        positive_embeddings = self.sat_embedding(data[1])
        negative_embeddings = self.sat_embedding(data[2])

        print(f"A:{anchor_embeddings}")
        print(f"P:{positive_embeddings}")
        print(f"N:{negative_embeddings}")
        # pairwise_dist = _pairwise_distances(embeddings)

        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]

    def get_config(self):
        return {self.siamese_network.get_config()}

    def load(self, filepath: str):
        self.siamese_network.load_weights(filepath)
