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
import pandas as pd
import sklearn
import sklearn.metrics
import os
import random
import statistics

from training_module.util import (
    get_data_from_tf,
    load_sample_dataset,
    most_frequent,
    load_loss_func,
)


def val(
    img_features,
    slide_label,
    c_model,
    args,
):
    """_summary_

    Args:
        img_features (_type_): _description_
        slide_label (_type_): _description_
        c_model (_type_): _description_
        args (_type_): _description_

    Returns:
        _type_: _description_
    """
    i_loss_func, b_loss_func = load_loss_func(
        args=args,
    )

    if args.gpu:
        gpus = tf.config.list_logical_devices("GPU")
        strategy = tf.distribute.MirroredStrategy(gpus)
        with strategy.scope():
            c_model_dict = c_model.call(img_features, slide_label)
            predict_slide_label = c_model_dict["predict_slide_label"]

            i_loss = i_loss_func(
                c_model_dict["ins_labels"],
                c_model_dict["ins_logits"],
            )
            I_Loss = tf.math.reduce_mean(i_loss)
            if args.mut_ex:
                I_Loss = I_Loss / args.n_class
            B_Loss = b_loss_func(c_model_dict["Y_true"], c_model_dict["Y_prob"])
            T_Loss = args.c1 * B_Loss + args.c2 * I_Loss
    else:
        c_model_dict = c_model.call(img_features, slide_label)
        predict_slide_label = c_model_dict["predict_slide_label"]

        i_loss = i_loss_func(
            c_model_dict["ins_labels"],
            c_model_dict["ins_logits"],
        )
        I_Loss = tf.math.reduce_mean(i_loss)
        if args.mut_ex:
            I_Loss = I_Loss / args.n_class
        B_Loss = b_loss_func(c_model_dict["Y_true"], c_model_dict["Y_prob"])
        T_Loss = args.c1 * B_Loss + args.c2 * I_Loss

    return I_Loss, B_Loss, T_Loss, predict_slide_label


def val_step(
    c_model,
    args,
):
    """_summary_

    Args:
        c_model (_type_): _description_
        args (_type_): _description_

    Returns:
        _type_: _description_
    """
    loss_t = list()
    loss_i = list()
    loss_b = list()

    slide_true_label = list()
    slide_predict_label = list()

    val_sample_dataset = load_sample_dataset(args=args, sample_name="val")
    features, labels = (
        val_sample_dataset["image_features"],
        val_sample_dataset["slide_labels"],
    )
    for i in range(len(labels)):
        print("=", end="")
        I_Loss, B_Loss, T_Loss, predict_slide_label = val(
            img_features=features[i],
            slide_label=labels[i],
            c_model=c_model,
            args=args,
        )

        loss_t.append(float(T_Loss))
        loss_i.append(float(I_Loss))
        loss_b.append(float(B_Loss))

        slide_true_label.append(labels[i])
        slide_predict_label.append(predict_slide_label)

    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(
        slide_true_label, slide_predict_label
    ).ravel()
    val_tn = int(tn)
    val_fp = int(fp)
    val_fn = int(fn)
    val_tp = int(tp)

    val_sensitivity = round(val_tp / (val_tp + val_fn), 2)
    val_specificity = round(val_tn / (val_tn + val_fp), 2)
    val_acc = round((val_tp + val_tn) / (val_tn + val_fp + val_fn + val_tp), 2)

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(
        slide_true_label, slide_predict_label, pos_label=1
    )
    val_auc = round(sklearn.metrics.auc(fpr, tpr), 2)

    val_loss = statistics.mean(loss_t)
    val_ins_loss = statistics.mean(loss_i)
    val_bag_loss = statistics.mean(loss_b)

    return {
        "val_loss": val_loss,
        "val_ins_loss": val_ins_loss,
        "val_bag_loss": val_bag_loss,
        "val_tn": val_tn,
        "val_fp": val_fp,
        "val_fn": val_fn,
        "val_tp": val_tp,
        "val_sensitivity": val_sensitivity,
        "val_specificity": val_specificity,
        "val_acc": val_acc,
        "val_auc": val_auc,
    }
