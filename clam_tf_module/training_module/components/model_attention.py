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


## Non-Gated Attention Network
class NG_Att_Net(tf.keras.Model):
    """_summary_

    Args:
        tf (_type_): _description_
    """

    def __init__(
        self,
        args,
        dim_features=1024,
        n_hidden_units=256,
    ):
        """_summary_

        Args:
            args (_type_): _description_
            dim_features (int, optional): _description_. Defaults to 1024.
            n_hidden_units (int, optional): _description_. Defaults to 256.
        """
        super(NG_Att_Net, self).__init__()
        self.args = args
        self.dim_features = dim_features
        self.n_hidden_units = n_hidden_units

        self.compression_model = tf.keras.models.Sequential()
        self.model = tf.keras.models.Sequential()

        self.fc_compress_layer = tf.keras.layers.Dense(
            units=self.args.dim_compress_features,
            activation="relu",
            input_shape=(self.dim_features,),
            kernel_initializer="glorot_normal",
            bias_initializer="zeros",
            name="Fully_Connected_Layer",
        )

        self.compression_model.add(self.fc_compress_layer)

        self.att_layer1 = tf.keras.layers.Dense(
            units=n_hidden_units,
            activation="linear",
            input_shape=(self.args.dim_compress_features,),
            kernel_initializer="glorot_normal",
            bias_initializer="zeros",
            name="Attention_layer1",
        )

        self.att_layer2 = tf.keras.layers.Dense(
            units=n_hidden_units,
            activation="tanh",
            input_shape=(self.args.dim_compress_features,),
            kernel_initializer="glorot_normal",
            bias_initializer="zeros",
            name="Attention_Layer2",
        )

        self.att_layer3 = tf.keras.layers.Dense(
            units=self.args.n_class,
            activation="linear",
            input_shape=(n_hidden_units,),
            kernel_initializer="glorot_normal",
            bias_initializer="zeros",
            name="Attention_Layer3",
        )

        self.model.add(self.att_layer1)
        self.model.add(self.att_layer2)

        if self.args.dropout_rate > 0.0:
            self.model.add(
                tf.keras.layers.Dropout(self.args.dropout_rate, name="Dropout_Layer")
            )

        self.model.add(self.att_layer3)

    def att_model(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        attention_model = [self.compression_model, self.model]
        return attention_model

    def call(self, img_features):
        """_summary_

        Args:
            img_features (_type_): _description_

        Returns:
            _type_: _description_
        """
        h = list()
        A = list()

        for i in img_features:
            c_imf = self.att_model()[0](i)
            h.append(c_imf)

        for j in h:
            a = self.att_model()[1](j)
            A.append(a)

        return {"h": h, "A": A}


## Gated Attention Network
class G_Att_Net(tf.keras.Model):
    """_summary_

    Args:
        tf (_type_): _description_
    """

    def __init__(
        self,
        args,
        dim_features=1024,
        n_hidden_units=256,
    ):
        """_summary_

        Args:
            args (_type_): _description_
            dim_features (int, optional): _description_. Defaults to 1024.
            n_hidden_units (int, optional): _description_. Defaults to 256.
        """
        super(G_Att_Net, self).__init__()
        self.args = args
        self.dim_features = dim_features
        self.n_hidden_units = n_hidden_units

        self.compression_model = tf.keras.models.Sequential()
        self.model_v = tf.keras.models.Sequential()
        self.model_u = tf.keras.models.Sequential()
        self.model = tf.keras.models.Sequential()

        self.fc_compress_layer = tf.keras.layers.Dense(
            units=self.args.dim_compress_features,
            activation="relu",
            input_shape=(self.dim_features,),
            kernel_initializer="glorot_normal",
            bias_initializer="zeros",
            name="Fully_Connected_Layer",
        )

        self.compression_model.add(self.fc_compress_layer)

        self.att_v_layer1 = tf.keras.layers.Dense(
            units=n_hidden_units,
            activation="linear",
            input_shape=(self.args.dim_compress_features,),
            kernel_initializer="glorot_normal",
            bias_initializer="zeros",
            name="Attention_V_Layer1",
        )

        self.att_v_layer2 = tf.keras.layers.Dense(
            units=n_hidden_units,
            activation="tanh",
            input_shape=(self.args.dim_compress_features,),
            kernel_initializer="glorot_normal",
            bias_initializer="zeros",
            name="Attention_V_Layer2",
        )

        self.att_u_layer1 = tf.keras.layers.Dense(
            units=n_hidden_units,
            activation="linear",
            input_shape=(self.args.dim_compress_features,),
            kernel_initializer="glorot_normal",
            bias_initializer="zeros",
            name="Attention_U_Layer1",
        )

        self.att_u_layer2 = tf.keras.layers.Dense(
            units=n_hidden_units,
            activation="sigmoid",
            input_shape=(self.args.dim_compress_features,),
            kernel_initializer="glorot_normal",
            bias_initializer="zeros",
            name="Attention_U_Layer2",
        )

        self.att_layer_f = tf.keras.layers.Dense(
            units=self.args.n_class,
            activation="linear",
            input_shape=(n_hidden_units,),
            kernel_initializer="glorot_normal",
            bias_initializer="zeros",
            name="Attention_Gated_Final_Layer",
        )

        self.model_v.add(self.att_v_layer1)
        self.model_v.add(self.att_v_layer2)

        self.model_u.add(self.att_u_layer1)
        self.model_u.add(self.att_u_layer2)

        if self.args.dropout_rate > 0.0:
            self.model_v.add(
                tf.keras.layers.Dropout(self.args.dropout_rate, name="Dropout_V_Layer")
            )
            self.model_u.add(
                tf.keras.layers.Dropout(self.args.dropout_rate, name="Dropout_U_Layer")
            )

        self.model.add(self.att_layer_f)

    def att_model(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        attention_model = [
            self.compression_model,
            self.model_v,
            self.model_u,
            self.model,
        ]
        return attention_model

    def call(self, img_features):
        """_summary_

        Args:
            img_features (_type_): _description_

        Returns:
            _type_: _description_
        """
        h = list()
        A = list()

        for i in img_features:
            c_imf = self.att_model()[0](i)
            h.append(c_imf)

        for j in h:
            att_v_output = self.att_model()[1](j)
            att_u_output = self.att_model()[2](j)
            att_input = tf.math.multiply(att_v_output, att_u_output)
            a = self.att_model()[3](att_input)
            A.append(a)

        return {"h": h, "A": A}
