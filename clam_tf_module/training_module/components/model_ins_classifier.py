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


## Instance-Level Classifier
class Ins(tf.keras.Model):
    """_summary_

    Args:
        tf (_type_): _description_
    """

    def __init__(
        self,
        args,
    ):
        """_summary_

        Args:
            args (_type_): _description_
        """
        super(Ins, self).__init__()
        self.args = args

        self.ins_model = list()
        self.m_ins_model = tf.keras.models.Sequential()
        self.m_ins_layer = tf.keras.layers.Dense(
            units=self.args.n_class,
            activation="linear",
            input_shape=(self.args.dim_compress_features,),
            name="Instance_Classifier_Layer",
        )
        self.m_ins_model.add(self.m_ins_layer)

        for i in range(self.args.n_class):
            self.ins_model.append(self.m_ins_model)

    def ins_classifier(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self.ins_model

    @staticmethod
    def generate_pos_labels(n_pos_sample):
        """_summary_

        Args:
            n_pos_sample (_type_): _description_

        Returns:
            _type_: _description_
        """
        return tf.fill(
            dims=[
                n_pos_sample,
            ],
            value=1,
        )

    @staticmethod
    def generate_neg_labels(n_neg_sample):
        """_summary_

        Args:
            n_neg_sample (_type_): _description_

        Returns:
            _type_: _description_
        """
        return tf.fill(
            dims=[
                n_neg_sample,
            ],
            value=0,
        )

    def in_call(self, n_ins, ins_classifier, h, A_I):
        """_summary_

        Args:
            n_ins (_type_): _description_
            ins_classifier (_type_): _description_
            h (_type_): _description_
            A_I (_type_): _description_

        Returns:
            _type_: _description_
        """
        pos_label = self.generate_pos_labels(n_ins)
        neg_label = self.generate_neg_labels(n_ins)
        ins_label_in = tf.concat(values=[pos_label, neg_label], axis=0)
        A_I = tf.reshape(tf.convert_to_tensor(A_I), (1, len(A_I)))

        top_pos_ids = tf.math.top_k(A_I, n_ins)[1][-1]
        pos_index = list()
        for i in top_pos_ids:
            pos_index.append(i)

        pos_index = tf.convert_to_tensor(pos_index)
        top_pos = list()
        for i in pos_index:
            top_pos.append(h[i])

        top_neg_ids = tf.math.top_k(-A_I, n_ins)[1][-1]
        neg_index = list()
        for i in top_neg_ids:
            neg_index.append(i)

        neg_index = tf.convert_to_tensor(neg_index)
        top_neg = list()
        for i in neg_index:
            top_neg.append(h[i])

        ins_in = tf.concat(values=[top_pos, top_neg], axis=0)
        logits_unnorm_in = list()
        logits_in = list()

        for i in range(self.args.n_class * n_ins):
            ins_score_unnorm_in = ins_classifier(ins_in[i])
            logit_in = tf.math.softmax(ins_score_unnorm_in)
            logits_unnorm_in.append(ins_score_unnorm_in)
            logits_in.append(logit_in)

        return ins_label_in, logits_unnorm_in, logits_in

    def out_call(self, n_ins, ins_classifier, h, A_O):
        """_summary_

        Args:
            n_ins (_type_): _description_
            ins_classifier (_type_): _description_
            h (_type_): _description_
            A_O (_type_): _description_

        Returns:
            _type_: _description_
        """
        # get compressed 512-dimensional instance-level feature vectors for following use, denoted by h
        A_O = tf.reshape(tf.convert_to_tensor(A_O), (1, len(A_O)))
        top_pos_ids = tf.math.top_k(A_O, n_ins)[1][-1]
        pos_index = list()
        for i in top_pos_ids:
            pos_index.append(i)

        pos_index = tf.convert_to_tensor(pos_index)
        top_pos = list()
        for i in pos_index:
            top_pos.append(h[i])

        # mutually-exclusive -> top k instances w/ highest attention scores ==> false pos = neg
        pos_ins_labels_out = self.generate_neg_labels(n_ins)
        ins_label_out = pos_ins_labels_out

        logits_unnorm_out = list()
        logits_out = list()

        for i in range(n_ins):
            ins_score_unnorm_out = ins_classifier(top_pos[i])
            logit_out = tf.math.softmax(ins_score_unnorm_out)
            logits_unnorm_out.append(ins_score_unnorm_out)
            logits_out.append(logit_out)

        return ins_label_out, logits_unnorm_out, logits_out

    def call(self, bag_label, h, A):
        """_summary_

        Args:
            bag_label (_type_): _description_
            h (_type_): _description_
            A (_type_): _description_

        Returns:
            _type_: _description_
        """
        n_ins = self.args.top_k_percent * len(h)
        n_ins = int(n_ins)
        # if n_ins computed above is less than 0, make n_ins be default be 8
        if n_ins == 0:
            n_ins += 8

        for i in range(self.args.n_class):
            ins_classifier = self.ins_classifier()[i]
            if i == bag_label:
                A_I = list()
                for j in range(len(A)):
                    a_i = A[j][0][i]
                    A_I.append(a_i)
                ins_label_in, logits_unnorm_in, logits_in = self.in_call(
                    n_ins, ins_classifier, h, A_I
                )
            else:
                if self.args.mut_ex:
                    A_O = list()
                    for j in range(len(A)):
                        a_o = A[j][0][i]
                        A_O.append(a_o)
                    ins_label_out, logits_unnorm_out, logits_out = self.out_call(
                        n_ins, ins_classifier, h, A_O
                    )
                else:
                    continue

        if self.args.mut_ex:
            ins_labels = tf.concat(values=[ins_label_in, ins_label_out], axis=0)
            ins_logits_unnorm = logits_unnorm_in + logits_unnorm_out
            ins_logits = logits_in + logits_out
        else:
            ins_labels = ins_label_in
            ins_logits_unnorm = logits_unnorm_in
            ins_logits = logits_in

        return {
            "ins_labels": ins_labels,
            "ins_logits_unnorm": ins_logits_unnorm,
            "ins_logits": ins_logits,
        }
