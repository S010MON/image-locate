import numpy as np
import tensorflow as tf

from keras import Model
from keras.applications import resnet
from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense, Dropout, Layer, Input, BatchNormalization
from keras import metrics


def vgg16_embedding(name: str) -> Model:
    vgg16 = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))

    # Freeze layers for training
    for layer in vgg16.layers:
        layer.trainable = False

    # Create a new 'top' of the model of fully-connected layers for Places365
    top_model = vgg16.output
    top_model = Flatten(name="flatten")(top_model)
    top_model = Dense(512, activation='relu', name="fc1")(top_model)
    top_model = Dropout(0.5, name="drop_fc1")(top_model)
    top_model = Dense(512, activation='relu', name="fc2")(top_model)
    top_model = Dropout(0.2, name="drop_fc2")(top_model)
    output_layer = Dense(256, activation='softmax', name="predictions")(top_model)

    embedding_model = Model(inputs=vgg16.input,
                            outputs=output_layer,
                            name=name)

    return embedding_model


def resnet_embedding(name: str) -> Model:
    base_cnn = resnet.ResNet50(
        weights="imagenet",
        input_shape=(200, 200, 3),
        include_top=False
    )

    flatten = Flatten()(base_cnn.output)
    dense1 = Dense(512, activation="relu")(flatten)
    dense1 = BatchNormalization()(dense1)
    dense2 = Dense(256, activation="relu")(dense1)
    dense2 = BatchNormalization()(dense2)
    output = Dense(256)(dense2)

    embedding = Model(base_cnn.input, output, name=name)

    trainable = False
    for layer in base_cnn.layers:
        if layer.name == "conv5_block1_out":
            trainable = True
        layer.trainable = trainable

    return embedding


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


class SiameseModel(Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, margin=0.5, base_network: str = 'resnet'):
        super().__init__()

        if base_network == 'resnet':
            input_shape = (200, 200)
            self.sat_embedding = resnet_embedding("Resnet50_Sat_Embedding")
            self.gnd_embedding = resnet_embedding("Resnet50_Gnd_Embedding")

        elif base_network == 'vgg16':
            input_shape = (224, 224)
            self.sat_embedding = vgg16_embedding("VGG16_Sat_Embedding")
            self.gnd_embedding = vgg16_embedding("VGG16_Gnd_Embedding")
        else:
            raise ValueError("Base network not selected.  Please choose from 'resnet' or 'vgg16' base networks")

        anchor_input = Input(name="anchor", shape=input_shape + (3,))
        positive_input = Input(name="positive", shape=input_shape + (3,))
        negative_input = Input(name="negative", shape=input_shape + (3,))

        distances = DistanceLayer()(
            self.gnd_embedding(resnet.preprocess_input(anchor_input)),
            self.sat_embedding(resnet.preprocess_input(positive_input)),
            self.sat_embedding(resnet.preprocess_input(negative_input)),
        )

        self.siamese_network = Model(
            inputs=[anchor_input, positive_input, negative_input], outputs=distances
        )
        self.margin = margin
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
        distance = tf.matmul(dp, tf.transpose(dn))

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
