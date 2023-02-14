# Copyright 2022 Mayo Clinic. All Rights Reserved.
#
# Author: Quincy Gu (M216613)
# Affliation: Division of Computational Pathology and Artificial Intelligence,
# Department of Laboratory Medicine and Pathology, Mayo Clinic College of Medicine and Science
# Email: Gu.Qiangqiang@mayo.edu
# Version: 1.0.1
# Created on: 11/28/2022 06:37 pm CST
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import tensorflow as tf
import numpy as np


## Single Bag-Level classifier
class S_Bag(tf.keras.Model):
    """_summary_

    Args:
        tf (_type_): _description_
    """

    def __init__(self, args,):
        """_summary_

        Args:
            args (_type_): _description_
        """
        super(S_Bag, self).__init__()
        self.args = args

        self.s_bag_model = tf.keras.models.Sequential()
        self.s_bag_layer = tf.keras.layers.Dense(
            units=1,
            activation="linear",
            input_shape=(self.args.n_class, self.args.dim_compress_features),
            name="Bag_Classifier_Layer",
        )
        self.s_bag_model.add(self.s_bag_layer)

    def bag_classifier(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self.s_bag_model

    def h_slide(self, A, h):
        """_summary_

        Args:
            A (_type_): _description_
            h (_type_): _description_

        Returns:
            _type_: _description_
        """
        # compute the slide-level representation aggregated per the attention score distribution for the mth class
        SAR = list()
        for i in range(len(A)):
            sar = tf.linalg.matmul(tf.transpose(A[i]), h[i])  # shape be (2,512)
            SAR.append(sar)
        slide_agg_rep = tf.math.add_n(SAR)  # return h_[slide,m], shape be (2,512)
        ## need to reshape slide_agg_rep be (1,2,512), which will be compatible with input layer dimension
        if len(slide_agg_rep.shape) == 2:
            slide_agg_rep = tf.reshape(slide_agg_rep, (1, slide_agg_rep.shape[0], slide_agg_rep.shape[1]))

        return slide_agg_rep

    def call(self, bag_label, A, h):
        """_summary_

        Args:
            bag_label (_type_): _description_
            A (_type_): _description_
            h (_type_): _description_

        Returns:
            _type_: _description_
        """
        slide_agg_rep = self.h_slide(A, h)
        bag_classifier = self.bag_classifier()
        slide_score_unnorm = bag_classifier(slide_agg_rep)
        slide_score_unnorm = tf.reshape(slide_score_unnorm, (1, self.args.n_class))
        Y_hat = tf.math.top_k(slide_score_unnorm, 1)[1][-1]
        Y_prob = tf.math.softmax(
            tf.reshape(slide_score_unnorm, (1, self.args.n_class))
        )  # shape be (1,2), predictions for each of the classes
        predict_slide_label = np.argmax(Y_prob.numpy())

        Y_true = tf.one_hot([bag_label], 2)

        return {
            "slide_score_unnorm": slide_score_unnorm,
            "Y_hat": Y_hat,
            "Y_prob": Y_prob,
            "predict_slide_label": predict_slide_label,
            "Y_true": Y_true,
        }


## Multiple Bag-Level classifiers (#classifiers == #classes)
class M_Bag(tf.keras.Model):
    """_summary_

    Args:
        tf (_type_): _description_
    """

    def __init__(self, args,):
        """_summary_

        Args:
            args (_type_): _description_
        """
        super(M_Bag, self).__init__()
        self.args = args

        self.m_bag_models = list()
        self.m_bag_model = tf.keras.models.Sequential()
        self.m_bag_layer = tf.keras.layers.Dense(
            units=1,
            activation="linear",
            input_shape=(1, self.args.dim_compress_features),
            name="Bag_Classifier_Layer",
        )
        self.m_bag_model.add(self.m_bag_layer)
        for i in range(self.args.n_class):
            self.m_bag_models.append(self.m_bag_model)

    def bag_classifier(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self.m_bag_models

    def h_slide(self, A, h):
        """_summary_

        Args:
            A (_type_): _description_
            h (_type_): _description_

        Returns:
            _type_: _description_
        """
        # compute the slide-level representation aggregated per the attention score distribution for the mth class
        SAR = list()
        for i in range(len(A)):
            sar = tf.linalg.matmul(tf.transpose(A[i]), h[i])  # shape be (2,512)
            SAR.append(sar)

        SAR_Branch = list()
        for i in range(self.args.n_class):
            sar_branch = list()
            for j in range(len(SAR)):
                sar_c = tf.reshape(SAR[j][i], (1, self.args.dim_compress_features))
                sar_branch.append(sar_c)
            SAR_Branch.append(sar_branch)

        slide_agg_rep = list()
        for k in range(self.args.n_class):
            slide_agg_rep.append(tf.math.add_n(SAR_Branch[k]))

        return slide_agg_rep

    def call(self, bag_label, A, h):
        """_summary_

        Args:
            bag_label (_type_): _description_
            A (_type_): _description_
            h (_type_): _description_

        Returns:
            _type_: _description_
        """
        slide_agg_rep = self.h_slide(A, h)

        # return s_[slide,m] (slide-level prediction scores)
        ssus = list()
        for i in range(self.args.n_class):
            bag_classifier = self.bag_classifier()[i]
            ssu = bag_classifier(slide_agg_rep[i])
            ssus.append(ssu[0][0])

        slide_score_unnorm = tf.convert_to_tensor(ssus)
        slide_score_unnorm = tf.reshape(slide_score_unnorm, (1, self.args.n_class))

        Y_hat = tf.math.top_k(slide_score_unnorm, 1)[1][-1]
        Y_prob = tf.math.softmax(slide_score_unnorm)
        predict_slide_label = np.argmax(Y_prob.numpy())

        Y_true = tf.one_hot([bag_label], 2)

        return {
            "slide_score_unnorm": slide_score_unnorm,
            "Y_hat": Y_hat,
            "Y_prob": Y_prob,
            "predict_slide_label": predict_slide_label,
            "Y_true": Y_true,
        }