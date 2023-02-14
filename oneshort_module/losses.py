import tensorflow as tf
import numpy as np


@tf.function
def triplet_loss(y_batch_train, logits):
    """
    Implementation of the triplet loss function
    Arguments:
    y_batch_train -- true labels.
    logits -- predictions
    Returns:
    loss -- real number, value of the loss
    """
    y_batch_train = tf.cast(tf.squeeze(y_batch_train), tf.float16)
    logits = tf.cast(tf.squeeze(logits), tf.float16)

    loss = tf.keras.losses.binary_crossentropy(y_batch_train, logits)
    # loss = tf.math.subtract(y_batch_train, logits)
    # loss = tf.math.reduce_mean(loss)
    loss = tf.reshape(loss, (1,))
    return loss
