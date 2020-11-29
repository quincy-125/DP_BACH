import tensorflow as tf
import numpy as np


class S_Bag(tf.keras.Model):
    def __init__(self, dim_compress_features=512, n_class=2):
        super(S_Bag, self).__init__()
        self.dim_compress_features = dim_compress_features
        self.n_class = n_class

        self.s_bag_model = tf.keras.models.Sequential()
        self.s_bag_layer = tf.keras.layers.Dense(
            units=1, activation='linear', input_shape=(self.n_class, self.dim_compress_features),
            name='Bag_Classifier_Layer'
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
            tf.reshape(slide_score_unnorm, (1, self.n_class)))  # shape be (1,2), predictions for each of the classes
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
            units=1, activation='linear', input_shape=(self.dim_compress_features,), name='Bag_Classifier_Layer'
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
        slide_agg_rep = tf.math.add_n(SAR)  # return h_[slide,m], shape be (2,512)

        return slide_agg_rep

    def in_call(self, bag_classifier, h_slide_I):
        ssu_in = bag_classifier(h_slide_I)[0][0]

        return ssu_in

    def out_call(self, bag_classifier, h_slide_O):
        ssu_out = bag_classifier(h_slide_O)[0][0]

        return ssu_out

    def call(self, bag_label, A, h):
        slide_agg_rep = self.h_slide(A, h)
        # unnormalized slide-level score (s_[slide,m]) with uninitialized entries, shape be (1,num_of_classes)
        slide_score_unnorm = tf.Variable(np.empty((1, self.n_class)), dtype=tf.float32)
        slide_score_unnorm = tf.reshape(slide_score_unnorm, (1, self.n_class)).numpy()

        # return s_[slide,m] (slide-level prediction scores)
        for i in range(self.n_class):
            bag_classifier = self.bag_classifier()[i]
            if i == bag_label:
                h_slide_I = tf.reshape(slide_agg_rep[i], (1, self.dim_compress_features))
                ssu_in = self.in_call(bag_classifier, h_slide_I)
            else:
                h_slide_O = tf.reshape(slide_agg_rep[i], (1, self.dim_compress_features))
                ssu_out = self.out_call(bag_classifier, h_slide_O)

        for i in range(self.n_class):
            if i == bag_label:
                slide_score_unnorm[0, i] = ssu_in
            else:
                slide_score_unnorm[0, i] = ssu_out
        slide_score_unnorm = tf.convert_to_tensor(slide_score_unnorm)

        Y_hat = tf.math.top_k(slide_score_unnorm, 1)[1][-1]
        Y_prob = tf.math.softmax(slide_score_unnorm)
        predict_slide_label = np.argmax(Y_prob.numpy())

        Y_true = tf.one_hot([bag_label], 2)

        return slide_score_unnorm, Y_hat, Y_prob, predict_slide_label, Y_true