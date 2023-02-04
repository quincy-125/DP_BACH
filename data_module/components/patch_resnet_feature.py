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

import sys

sys_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(sys_dir))

from components.patch_extract import slide_extract_patches
from components.utils import *

configure_logging(script_name="cross_val_split")


def customize_res50(
    res_weights="imagenet",
    res_layer="conv4_block1_0_conv",
    res_top=False,
    res_trainable=False,
    input_shape=(256, 256, 3),
):
    """_summary_

    Args:
        res_weights (str, optional): _description_. Defaults to "imagenet".
        res_layer (str, optional): _description_. Defaults to "conv4_block1_0_conv".
        res_top (bool, optional): _description_. Defaults to False.
        res_trainable (bool, optional): _description_. Defaults to False.
        input_shape (tuple, optional): _description_. Defaults to (256, 256, 3).

    Returns:
        _type_: _description_
    """
    ## load the ResNet50 model from https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/ResNet50
    resnet50_model = tf.keras.applications.resnet50.ResNet50(
        include_top=res_top, weights=res_weights, input_shape=input_shape
    )
    ## freeze training when is.trainable is False
    resnet50_model.trainable = res_trainable

    ## create a new resnet50 model based on original resnet50 model ended after the 3rd residual block with the layer_name be 'conv4_block1_0_conv'
    custom_res50_model = tf.keras.Model(
        inputs=resnet50_model.input, outputs=resnet50_model.get_layer(res_layer).output
    )

    ## add adaptive mean-spatial pooling after the new model
    adaptive_mean_spatial_layer = tf.keras.layers.GlobalAvgPool2D()

    return custom_res50_model, adaptive_mean_spatial_layer


def extract_patch_feature(custom_res50_model, adaptive_mean_spatial_layer, patch_path):
    if len(patch_path) > 1:
        if tf.io.gfile.isdir(patch_path):
            patches = []
