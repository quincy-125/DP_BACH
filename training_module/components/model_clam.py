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
# See the License for the specific language governing permissions andecho cuda
# limitations under the License.
# ==============================================================================


import tensorflow as tf

from training_module.components.model_attention import G_Att_Net, NG_Att_Net
from training_module.components.model_bag_classifier import S_Bag, M_Bag
from training_module.components.model_ins_classifier import Ins


## Single CLAM Model (Single Bag-Level Classifier)
class S_CLAM(tf.keras.Model):
    """_summary_

    Args:
        tf (_type_): _description_
    """

    def __init__(self, args,):
        """_summary_

        Args:
            args (_type_): _description_
        """
        super(S_CLAM, self).__init__()
        self.args = args

        self.net_shape_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        self.net_shape = self.net_shape_dict[self.args.net_size]

        if self.args.att_gate:
            self.att_net = G_Att_Net(
                args=self.args,
                dim_features=self.net_shape[0],
                n_hidden_units=self.net_shape[2],
            )
        else:
            self.att_net = NG_Att_Net(
                args=self.args,
                dim_features=self.net_shape[0],
                n_hidden_units=self.net_shape[2],
            )

        self.ins_net = Ins(args=self.args,)
        self.bag_net = S_Bag(args=self.args,)

    def networks(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        c_nets = {
            "a_net": self.att_net, 
            "i_net": self.ins_net, 
            "b_net": self.bag_net,
        }

        return c_nets

    def clam_model(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        att_model = self.att_net.att_model()
        ins_classifier = self.ins_net.ins_classifier()
        bag_classifier = self.bag_net.bag_classifier()

        clam_model = {
            "att_model": att_model, 
            "ins_classifier": ins_classifier, 
            "bag_classifier": bag_classifier,
        }

        return clam_model

    def call(self, img_features, slide_label):
        """_summary_

        Args:
            img_features (_type_): original 1024-dimensional instance-level feature vectors
            slide_label (_type_): ground-truth slide label, could be 0 or 1 for binary classification

        Returns:
            _type_: _description_
        """
        att_net_dict = self.att_net.call(img_features)
        (h, att_score) = (att_net_dict["h"], att_net_dict["A"])
        A = tf.math.softmax(att_score)  # softmax on attention scores

        if self.args.att_only:
            return att_score

        ins_net_dict = self.ins_net.call(slide_label, h, A)

        bag_net_dict = self.bag_net.call(slide_label, A, h)

        return {
            "att_score": att_score,
            "A": A,
            "h": h,
            "ins_labels": ins_net_dict["ins_labels"],
            "ins_logits_unnorm": ins_net_dict["ins_logits_unnorm"],
            "ins_logits": ins_net_dict["ins_logits"],
            "slide_score_unnorm": bag_net_dict["slide_score_unnorm"],
            "Y_prob": bag_net_dict["Y_prob"],
            "Y_hat": bag_net_dict["Y_hat"],
            "Y_true": bag_net_dict["Y_true"],
            "predict_slide_label": bag_net_dict["predict_slide_label"],
        }


## Multiple CLAM (Multiple Bag-Level Classifiers)
class M_CLAM(tf.keras.Model):
    """_summary_

    Args:
        tf (_type_): _description_
    """

    def __init__(self, args,):
        """_summary_

        Args:
            args (_type_): _description_
        """
        super(M_CLAM, self).__init__()
        self.args = args

        self.net_shape_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        self.net_shape = self.net_shape_dict[self.args.net_size]

        if self.att_gate:
            self.att_net = G_Att_Net(
                args=self.args,
                dim_features=self.net_shape[0],
                n_hidden_units=self.net_shape[2],
            )
        else:
            self.att_net = NG_Att_Net(
                args=self.args,
                dim_features=self.net_shape[0],
                n_hidden_units=self.net_shape[2],
            )

        self.ins_net = Ins(args=self.args,)
        self.bag_net = M_Bag(args=self.args,)

    def networks(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        c_nets = {
            "a_net": self.att_net, 
            "i_net": self.ins_net, 
            "b_net": self.bag_net,
        }

        return c_nets

    def clam_model(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        att_model = self.att_net.att_model()
        ins_classifier = self.ins_net.ins_classifier()
        bag_classifier = self.bag_net.bag_classifier()

        clam_model = {
            "att_model": att_model, 
            "ins_classifier": ins_classifier, 
            "bag_classifier": bag_classifier,
        }

        return clam_model

    def call(self, img_features, slide_label):
        """_summary_

        Args:
            img_features (_type_): original 1024-dimensional instance-level feature vectors
            slide_label (_type_): ground-truth slide label, could be 0 or 1 for binary classification

        Returns:
            _type_: _description_
        """
        att_net_dict = self.att_net.call(img_features)
        (h, att_score) = (att_net_dict["h"], att_net_dict["A"])
        A = tf.math.softmax(att_score)  # softmax on attention scores

        if self.args.att_only:
            return att_score

        ins_net_dict = self.ins_net.call(slide_label, h, A)

        bag_net_dict = self.bag_net.call(slide_label, A, h)

        return {
            "att_score": att_score,
            "A": A,
            "h": h,
            "ins_labels": ins_net_dict["ins_labels"],
            "ins_logits_unnorm": ins_net_dict["ins_logits_unnorm"],
            "ins_logits": ins_net_dict["ins_logits"],
            "slide_score_unnorm": bag_net_dict["slide_score_unnorm"],
            "Y_prob": bag_net_dict["Y_prob"],
            "Y_hat": bag_net_dict["Y_hat"],
            "Y_true": bag_net_dict["Y_true"],
            "predict_slide_label": bag_net_dict["predict_slide_label"],
        }