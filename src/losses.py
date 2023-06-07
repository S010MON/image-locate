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
    distance_an_min = tf.reduce_min(tf.where(distance_an > distance_ap[:, tf.newaxis], distance_an, tf.float32.max), axis=1)
    distance_an_min = tf.cast(distance_an_min, tf.float32)

    # Calculate the loss for each pair (batch_size,)
    loss = tf.maximum(distance_ap - distance_an_min + alpha, 0.0)

    return tf.reduce_mean(loss)


def soft_margin_triplet_loss(gnd_embedding: tf.Tensor,
                            sat_pos_embedding: tf.Tensor,
                            sat_neg_embedding: tf.Tensor,
                             loss_weight=1) -> tf.Tensor:
    """
    L(d_p, d_n) = 1 / (1 + exp(d_p - d_n))
    :param dist_pos: The Euclidean distance between the positive and anchor embeddings (batch_size, 1)
    :param dist_neg: The Euclidean distance between the negative and anchor embeddings (batch_size, 1)
    :return: a tf.Tensor of losses for each triplet in the batch (batch_size, 1)
    """

    # Create a (batch_size, 1) tensor of distances from anchor to positive
    distance_ap = tf.reduce_sum(tf.square(gnd_embedding - sat_pos_embedding), -1)

    # Create a (batch_size, batch_size) tensor of distances between all of the negatives
    distance_an = np.square(scipy.spatial.distance.cdist(gnd_embedding, sat_neg_embedding))

    # For each row of distance_an, select the minimum distance that is greater than the corresponding distance_ap entry
    # shape (batch_size,)
    distance_an_min = tf.reduce_min(tf.where(distance_an > distance_ap[:, tf.newaxis], distance_an, tf.float32.max), axis=1)
    distance_an_min = tf.cast(distance_an_min, tf.float32)

    # Calculate the loss for each pair (batch_size,)
    loss = tf.math.log(1 + tf.math.exp(distance_ap - distance_an_min))
    return tf.reduce_mean(loss)

    # # sat_matrix = tf.nn.l2_normalize(sat_cropped, axis=[2, 3, 4]) # Don't think this line is needed
    #
    # # Create a distance matrix shape=(batch_grd, batch_sat)
    # distance = 2 - 2 * tf.transpose(tf.reduce_sum(dist_pos * tf.expand_dims(dist_neg, axis=0), axis=[2, 3, 4]))
    #
    # pos_dist = tf.diag_part(dist_array)
    #
    # pair_n = batch_size * (batch_size - 1.0)
    #
    # # satellite to ground
    # triplet_dist_g2s = pos_dist - dist_array
    # loss_g2s = tf.reduce_sum(tf.log(1 + tf.exp(triplet_dist_g2s * loss_weight))) / pair_n
    #
    # # ground to satellite
    # triplet_dist_s2g = tf.expand_dims(pos_dist, 1) - dist_array
    # loss_s2g = tf.reduce_sum(tf.log(1 + tf.exp(triplet_dist_s2g * loss_weight))) / pair_n
    #
    # loss = (loss_g2s + loss_s2g) / 2.0


def weighted_ranking_triplet_loss(dist_pos: tf.Tensor, dist_neg: tf.Tensor, alpha=10.0) -> tf.Tensor:
    """
    L(d_p, d_n) = 1 / (1 + exp(alpha* (d_p - d_n)))
    :param dist_pos: The Euclidean distance between the positive and anchor embeddings (batch_size, 1)
    :param dist_neg: The Euclidean distance between the negative and anchor embeddings (batch_size, 1)
    :param alpha: the weighting value
    :return: a tf.Tensor of losses for each triplet in the batch (batch_size, 1)
    """
    return tf.math.log(1 + tf.math.exp(alpha * (dist_pos - dist_neg)))
