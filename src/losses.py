import tensorflow as tf


def max_margin_triplet_loss(dist_pos: tf.Tensor, dist_neg: tf.Tensor, alpha: float = 0.5) -> tf.Tensor:
    """
    L(d_p, d_n) = max( d_p - d_n + alpha, 0)
    :param dist_pos: The Euclidean distance between the positive and anchor embeddings (batch_size, 1)
    :param dist_neg: The Euclidean distance between the negative and anchor embeddings (batch_size, 1)
    :param alpha: the margin
    :return: a tf.Tensor of losses for each triplet in the batch (batch_size, 1)
    """
    return tf.maximum(dist_pos - dist_neg + alpha, 0.0)


def soft_margin_triplet_loss(dist_pos: tf.Tensor, dist_neg: tf.Tensor, loss_weight=1) -> tf.Tensor:
    """
    L(d_p, d_n) = 1 / (1 + exp(d_p - d_n))
    :param dist_pos: The Euclidean distance between the positive and anchor embeddings (batch_size, 1)
    :param dist_neg: The Euclidean distance between the negative and anchor embeddings (batch_size, 1)
    :return: a tf.Tensor of losses for each triplet in the batch (batch_size, 1)
    """

    # return tf.math.log(1 + tf.math.exp(dist_pos - dist_neg))

    # sat_matrix = tf.nn.l2_normalize(sat_cropped, axis=[2, 3, 4]) # Don't think this line is needed

    # Create a distance matrix shape=(batch_grd, batch_sat)
    distance = 2 - 2 * tf.transpose(tf.reduce_sum(dist_pos * tf.expand_dims(dist_neg, axis=0), axis=[2, 3, 4]))

    pos_dist = tf.diag_part(dist_array)

    pair_n = batch_size * (batch_size - 1.0)

    # satellite to ground
    triplet_dist_g2s = pos_dist - dist_array
    loss_g2s = tf.reduce_sum(tf.log(1 + tf.exp(triplet_dist_g2s * loss_weight))) / pair_n

    # ground to satellite
    triplet_dist_s2g = tf.expand_dims(pos_dist, 1) - dist_array
    loss_s2g = tf.reduce_sum(tf.log(1 + tf.exp(triplet_dist_s2g * loss_weight))) / pair_n

    loss = (loss_g2s + loss_s2g) / 2.0


def weighted_ranking_triplet_loss(dist_pos: tf.Tensor, dist_neg: tf.Tensor, alpha=10.0) -> tf.Tensor:
    """
    L(d_p, d_n) = 1 / (1 + exp(alpha* (d_p - d_n)))
    :param dist_pos: The Euclidean distance between the positive and anchor embeddings (batch_size, 1)
    :param dist_neg: The Euclidean distance between the negative and anchor embeddings (batch_size, 1)
    :param alpha: the weighting value
    :return: a tf.Tensor of losses for each triplet in the batch (batch_size, 1)
    """
    return tf.math.log(1 + tf.math.exp(alpha * (dist_pos - dist_neg)))
