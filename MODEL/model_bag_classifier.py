import tensorflow as tf
import numpy as np


class S_Bag(tf.keras.Model):
    def __init__(self, dim_compress_features=512, n_class=2):
        super(S_Bag, self).__init__()
        self.dim_compress_features = dim_compress_features
        self.n_class = n_class

        self.s_bag_model = tf.keras.models.Sequential()
        self.s_bag_layer = tf.keras.layers.Dense(
            units=1,
            activation="linear",
            input_shape=(self.n_class, self.dim_compress_features),
            name="Bag_Classifier_Layer",
        )
        self.s_bag_model.add(self.s_bag_layer)

    def bag_classifier(self):
        return self.s_bag_model

    def h_slide(self, A, h):
        # compute the slide-level representation aggregated per the attention score distribution for the mth class
        SAR = list()
        for i in range(len(A)):
            sar = tf.linalg.matmul(tf.transpose(A[i]), h[i])  # shape be (2,512)
            SAR.append(sar)
        slide_agg_rep = tf.math.add_n(SAR)  # return h_[slide,m], shape be (2,512)

        return slide_agg_rep

    def call(self, bag_label, A, h):
        slide_agg_rep = self.h_slide(A, h)
        bag_classifier = self.bag_classifier()
        slide_score_unnorm = bag_classifier(slide_agg_rep)
        slide_score_unnorm = tf.reshape(slide_score_unnorm, (1, self.n_class))
        Y_hat = tf.math.top_k(slide_score_unnorm, 1)[1][-1]
        Y_prob = tf.math.softmax(
            tf.reshape(slide_score_unnorm, (1, self.n_class))
        )  # shape be (1,2), predictions for each of the classes
        predict_slide_label = np.argmax(Y_prob.numpy())

        Y_true = tf.one_hot([bag_label], 2)

        return slide_score_unnorm, Y_hat, Y_prob, predict_slide_label, Y_true


class M_Bag(tf.keras.Model):
    def __init__(self, dim_compress_features=512, n_class=2):
        super(M_Bag, self).__init__()
        self.dim_compress_features = dim_compress_features
        self.n_class = n_class

        self.m_bag_models = list()
        self.m_bag_model = tf.keras.models.Sequential()
        self.m_bag_layer = tf.keras.layers.Dense(
            units=1,
            activation="linear",
            input_shape=(1, self.dim_compress_features),
            name="Bag_Classifier_Layer",
        )
        self.m_bag_model.add(self.m_bag_layer)
        for i in range(self.n_class):
            self.m_bag_models.append(self.m_bag_model)

    def bag_classifier(self):
        return self.m_bag_models

    def h_slide(self, A, h):
        # compute the slide-level representation aggregated per the attention score distribution for the mth class
        SAR = list()
        for i in range(len(A)):
            sar = tf.linalg.matmul(tf.transpose(A[i]), h[i])  # shape be (2,512)
            SAR.append(sar)

        SAR_Branch = list()
        for i in range(self.n_class):
            sar_branch = list()
            for j in range(len(SAR)):
                sar_c = tf.reshape(SAR[j][i], (1, self.dim_compress_features))
                sar_branch.append(sar_c)
            SAR_Branch.append(sar_branch)

        slide_agg_rep = list()
        for k in range(self.n_class):
            slide_agg_rep.append(tf.math.add_n(SAR_Branch[k]))

        return slide_agg_rep

    def call(self, bag_label, A, h):
        slide_agg_rep = self.h_slide(A, h)

        # return s_[slide,m] (slide-level prediction scores)
        ssus = list()
        for i in range(self.n_class):
            bag_classifier = self.bag_classifier()[i]
            ssu = bag_classifier(slide_agg_rep[i])
            ssus.append(ssu[0][0])

        slide_score_unnorm = tf.convert_to_tensor(ssus)
        slide_score_unnorm = tf.reshape(slide_score_unnorm, (1, self.n_class))

        Y_hat = tf.math.top_k(slide_score_unnorm, 1)[1][-1]
        Y_prob = tf.math.softmax(slide_score_unnorm)
        predict_slide_label = np.argmax(Y_prob.numpy())

        Y_true = tf.one_hot([bag_label], 2)

        return slide_score_unnorm, Y_hat, Y_prob, predict_slide_label, Y_true
