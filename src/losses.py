import tensorflow as tf


def hard_margin_triplet_loss(dist_pos: tf.Tensor, dist_neg: tf.Tensor, alpha: float = 0.5) -> tf.Tensor:
    """
    L(d_p, d_n) = max( d_p - d_n + alpha, 0)
    :param dist_pos: The Euclidean distance between the positive and anchor embeddings (batch_size, 1)
    :param dist_neg: The Euclidean distance between the negative and anchor embeddings (batch_size, 1)
    :param alpha: the margin
    :return: a tf.Tensor of losses for each triplet in the batch (batch_size, 1)
    """
    return tf.maximum(dist_pos - dist_neg + alpha, 0.0)


def distance_based_logistic_loss(dist_pos: tf.Tensor, dist_neg: tf.Tensor) -> tf.Tensor:
    """
    L(d_p, d_n) = 1 / (1 + exp(d_p - d_n))
    :param dist_pos: The Euclidean distance between the positive and anchor embeddings (batch_size, 1)
    :param dist_neg: The Euclidean distance between the negative and anchor embeddings (batch_size, 1)
    :param alpha: the margin
    :return: a tf.Tensor of losses for each triplet in the batch (batch_size, 1)
    """
    return tf.math.log(1 + tf.math.exp(dist_pos - dist_neg))
