import tensorflow as tf
from tensorflow.keras.layers import Layer


class LosslessTripletLossLayer(Layer):
    # https://gist.github.com/marcolivierarsenault/a7ef5ab45e1fbb37fbe13b37a0de0257
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        self.neg_loss = None
        self.pos_loss = None
        self.neg_hist = None
        self.pos_hist = None
        self.loss = None
        self.embedding_size = 128
        super(LosslessTripletLossLayer, self).__init__(**kwargs)

    def triplet_loss(self, inputs):
        N = self.embedding_size
        beta = N
        epsilon = 1.00000008
        anchor, positive, negative = inputs
        p_dist = tf.math.reduce_sum(tf.math.square(anchor - positive), axis=-1)
        n_dist = tf.math.reduce_sum(tf.math.square(anchor - negative), axis=-1)
        # self.neg_dist = -tf.math.log(-tf.divide((beta - n_dist), beta) + epsilon)
        # self.pos_dist = -tf.math.log(-tf.divide(p_dist, beta) + epsilon)
        # p_dist = tf.keras.backend.sum(tf.keras.backend.square(anchor - positive), axis=-1)
        # n_dist = tf.keras.backend.sum(tf.keras.backend.square(anchor - negative), axis=-1)
        self.neg_dist = n_dist
        self.pos_dist = p_dist
        self.neg_hist = n_dist
        self.pos_hist = p_dist
        res = tf.math.reduce_sum(
            tf.math.maximum(p_dist - n_dist + self.alpha, 0), axis=0
        )
        # res = tf.keras.backend.sum(tf.keras.backend.max(p_dist - n_dist + self.alpha, 0), axis=0)
        return res

    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        self.loss = loss
        return self.loss, self.neg_dist, self.pos_dist, self.neg_hist, self.pos_hist
