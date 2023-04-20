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

from training_module.util import (
    load_sample_dataset,
    load_loss_func,
)


def forward_val(c_model, args):
    """_summary_

    Args:
        c_model (_type_): _description_
        args (_type_): _description_

    Returns:
        _type_: _description_
    """
    val_sample_dataset = load_sample_dataset(args=args, sample_name="val")
    features, labels = (
        val_sample_dataset["image_features"],
        val_sample_dataset["slide_labels"],
    )

    ins_labels = list()
    ins_logits = list()
    y_true = list()
    y_prob = list()
    pred_labels = list()

    for i in range(len(labels)):
        c_model_dict = c_model.call(features[i], labels[i])
        ins_labels.append(c_model_dict["ins_labels"])
        ins_logits.append(c_model_dict["ins_logits"])
        y_true.append(c_model_dict["Y_true"])
        y_prob.append(c_model_dict["Y_prob"])
        pred_labels.append(c_model_dict["predict_slide_label"])

    ins_labels = tf.convert_to_tensor(sum(ins_labels) / len(ins_labels))
    ins_logits = tf.convert_to_tensor(sum(ins_logits) / len(ins_logits))
    y_true = tf.convert_to_tensor(sum(y_true) / len(y_true))
    y_prob = tf.convert_to_tensor(sum(y_prob) / len(y_prob))

    outputs_dict = {
        "ins_labels": ins_labels,
        "ins_logits": ins_logits,
        "y_true": y_true,
        "y_prob": y_prob,
        "true_labels": labels,
        "pred_labels": pred_labels,
    }
    return outputs_dict


def backward_val(
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
    i_loss_func, b_loss_func = load_loss_func(
        args=args,
    )

    if args.gpu:
        gpus = tf.config.list_logical_devices("GPU")
        strategy = tf.distribute.MirroredStrategy(gpus)
        with strategy.scope():
            outputs_dict = forward_val(c_model=c_model, args=args)
            i_loss = i_loss_func(outputs_dict["ins_labels"], outputs_dict["ins_logits"])
            i_loss = tf.math.reduce_mean(i_loss)
            if args.mut_ex:
                i_loss = i_loss / args.n_class
            b_loss = b_loss_func(outputs_dict["y_true"], outputs_dict["y_prob"])
            t_loss = args.c1 * b_loss + args.c2 * i_loss
    else:
        outputs_dict = forward_val(c_model=c_model, args=args)
        i_loss = i_loss_func(outputs_dict["ins_labels"], outputs_dict["ins_logits"])
        i_loss = tf.math.reduce_mean(i_loss)
        if args.mut_ex:
            i_loss = i_loss / args.n_class
        b_loss = b_loss_func(outputs_dict["y_true"], outputs_dict["y_prob"])
        t_loss = args.c1 * b_loss + args.c2 * i_loss

    return i_loss, b_loss, t_loss


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
    outputs_dict = forward_val(c_model=c_model, args=args)
    i_loss, b_loss, t_loss = backward_val(c_model=c_model, args=args)

    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(
        outputs_dict["true_labels"], outputs_dict["pred_labels"]
    ).ravel()
    val_tn, val_fp, val_fn, val_tp = int(tn), int(fp), int(fn), int(tp)

    val_sensitivity = round(val_tp / (val_tp + val_fn), 2)
    val_specificity = round(val_tn / (val_tn + val_fp), 2)
    val_acc = round((val_tp + val_tn) / (val_tn + val_fp + val_fn + val_tp), 2)

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(
        outputs_dict["true_labels"], outputs_dict["pred_labels"], pos_label=1
    )
    val_auc = round(sklearn.metrics.auc(fpr, tpr), 2)

    return {
        "val_loss": t_loss,
        "val_ins_loss": i_loss,
        "val_bag_loss": b_loss,
        "val_sensitivity": val_sensitivity,
        "val_specificity": val_specificity,
        "val_acc": val_acc,
        "val_auc": val_auc,
    }
