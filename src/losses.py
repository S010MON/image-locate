import numpy as np
import scipy
import tensorflow as tf


def max_margin_triplet_loss(gnd_embedding: tf.Tensor,
                            sat_pos_embedding: tf.Tensor,
                            sat_neg_embedding: tf.Tensor,
                            alpha: float = 0.5) -> tf.Tensor:
    """
    L(d_p, d_n) = max( d_p - d_n + alpha, 0)
    :param gnd_embedding: (batch_size, dims)
    :param sat_pos_embedding: (batch_size, dims)
    :param sat_neg_embedding: (batch_size, dims)
    :param alpha: the margin
    :return: a tf.Tensor of losses for each triplet in the batch (batch_size, 1)
    """
    # Create a (batch_size, 1) tensor of distances from anchor to positive
    distance_ap = tf.reduce_sum(tf.square(gnd_embedding - sat_pos_embedding), -1)

    # Create a (batch_size, batch_size) tensor of distances between all of the negatives
    distance_an = np.square(scipy.spatial.distance.cdist(gnd_embedding, sat_neg_embedding))

    # For each row of distance_an, select the minimum distance that is greater than the corresponding distance_ap entry
    # shape (batch_size,)
    distance_an_min = tf.reduce_min(tf.where(distance_an > distance_ap[:, tf.newaxis], distance_an, tf.float32.max),
                                    axis=1)
    distance_an_min = tf.cast(distance_an_min, tf.float32)

    # Calculate the loss for each pair (batch_size,)
    loss = tf.maximum(distance_ap - distance_an_min + alpha, 0.0)

    return tf.reduce_mean(loss)


def soft_margin_triplet_loss(gnd_embedding: tf.Tensor,
                             sat_pos_embedding: tf.Tensor,
                             sat_neg_embedding: tf.Tensor,
                             alpha=10.0) -> tf.Tensor:
    """
    L(d_p, d_n) = 1 / (1 + exp(d_p - d_n))
    :param gnd_embedding: (batch_size, dims)
    :param sat_pos_embedding: (batch_size, dims)
    :param sat_neg_embedding: (batch_size, dims)
    :param alpha: the coefficient of the weight of the loss function
    :return: a tf.Tensor of losses for each triplet in the batch (batch_size, 1)
    """

    # Create a (batch_size, 1) tensor of distances from anchor to positive
    distance_ap = tf.reduce_sum(tf.square(gnd_embedding - sat_pos_embedding), -1)

    # Create a (batch_size, batch_size) tensor of distances between all of the negatives
    distance_an = np.square(scipy.spatial.distance.cdist(gnd_embedding, sat_neg_embedding))

    batch_size = gnd_embedding.shape[0]
    pair_n = batch_size * (batch_size - 1.0)

    # Ground to satellite
    triplet_dist_gnd2sat = distance_ap - distance_an
    loss_gnd2sat = tf.reduce_sum(tf.math.log(1 + tf.math.exp(alpha * triplet_dist_gnd2sat))) / pair_n

    # Satellite to ground
    triplet_dist_sat2gnd = tf.expand_dims(distance_ap, 1) - distance_an
    loss_sat2gnd = tf.reduce_sum(tf.math.log(1 + tf.math.exp(alpha * triplet_dist_sat2gnd))) / pair_n

    loss = (loss_sat2gnd + loss_gnd2sat) / 2.0
    return loss
