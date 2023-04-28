import tensorflow as tf

from keras import Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.layers import Flatten, Dense, Dropout, Layer, Input
from keras import metrics


def init_embedding_model(name: str) -> Model:
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


def init_network_model(batch_size, input_shape=(224, 224, 3)) -> Model:
    anchor_input = Input(name="anchor",
                         batch_size=batch_size,
                         shape=input_shape)

    positive_input = Input(name="positive",
                           batch_size=batch_size,
                           shape=input_shape)

    negative_input = Input(name="negative",
                           batch_size=batch_size,
                           shape=input_shape)

    sat_model = init_embedding_model("satellite")
    ter_model = init_embedding_model("terrestrial")

    distances = DistanceLayer()(
        sat_model(preprocess_input(anchor_input)),
        ter_model(preprocess_input(positive_input)),
        ter_model(preprocess_input(negative_input)),
    )

    return Model(
        inputs=[anchor_input, positive_input, negative_input],
        outputs=distances
    )


class SiameseNetwork(Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    Siamese Network.
    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)

    source: https://keras.io/examples/vision/siamese_network/
    """

    def __init__(self, siamese_model=None, batch_size=32, margin=0.5):
        super().__init__()
        if siamese_model is None:
            self.siamese_model = init_network_model(batch_size)
        else:
            self.siamese_model = siamese_model
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")


    def call(self, inputs):
        return self.siamese_model(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_model.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_model.trainable_weights)
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
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_model(data)

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

    def save(self, file_path: str):
        self.siamese_model.save(file_path)

