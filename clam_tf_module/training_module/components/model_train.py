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
import sklearn
import sklearn.metrics
import os
from statistics import mean

from training_module.util import (
    load_sample_dataset,
    load_optimizers,
    load_loss_func,
)

os.environ["HYDRA_FULL_ERROR"] = "1"


def forward_propagation(c_model, args):
    """_summary_

    Args:
        c_model (_type_): _description_
        args (_type_): _description_

    Returns:
        _type_: _description_
    """
    train_sample_dataset = load_sample_dataset(args=args, sample_name="train")
    features, labels = (
        train_sample_dataset["image_features"],
        train_sample_dataset["slide_labels"],
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


def backward_propagation(
    c_model,
    args,
):
    """å

    Args:
        c_model (_type_): _description_
        args (_type_): _description_

    Returns:
        _type_: _description_
    """
    i_optimizer, b_optimizer, a_optimizer = load_optimizers(
        args=args,
    )
    i_loss_func, b_loss_func = load_loss_func(
        args=args,
    )

    if args.gpu:
        gpus = tf.config.list_logical_devices("GPU")
        strategy = tf.distribute.MirroredStrategy(gpus)

        with strategy.scope():
            with tf.GradientTape() as i_tape, tf.GradientTape() as b_tape, tf.GradientTape() as a_tape:
                outputs_dict = forward_propagation(c_model=c_model, args=args)
                i_loss = i_loss_func(
                    outputs_dict["ins_labels"], outputs_dict["ins_logits"]
                )
                i_loss = tf.math.reduce_mean(i_loss)
                if args.mut_ex:
                    i_loss = i_loss / args.n_class
                b_loss = b_loss_func(outputs_dict["y_true"], outputs_dict["y_prob"])
                t_loss = args.c1 * b_loss + args.c2 * i_loss
    else:
        with tf.GradientTape() as i_tape, tf.GradientTape() as b_tape, tf.GradientTape() as a_tape:
            outputs_dict = forward_propagation(c_model=c_model, args=args)
            i_loss = i_loss_func(outputs_dict["ins_labels"], outputs_dict["ins_logits"])
            i_loss = tf.math.reduce_mean(i_loss)
            if args.mut_ex:
                i_loss = i_loss / args.n_class
            b_loss = b_loss_func(outputs_dict["y_true"], outputs_dict["y_prob"])
            t_loss = args.c1 * b_loss + args.c2 * i_loss

    i_grad = i_tape.gradient(i_loss, c_model.networks()["i_net"].trainable_weights)
    i_optimizer.apply_gradients(
        zip(i_grad, c_model.networks()["i_net"].trainable_weights)
    )

    b_grad = b_tape.gradient(b_loss, c_model.networks()["b_net"].trainable_weights)
    b_optimizer.apply_gradients(
        zip(b_grad, c_model.networks()["b_net"].trainable_weights)
    )

    a_grad = a_tape.gradient(t_loss, c_model.networks()["a_net"].trainable_weights)
    a_optimizer.apply_gradients(
        zip(a_grad, c_model.networks()["a_net"].trainable_weights)
    )

    return i_loss, b_loss, t_loss


def train_step(
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
    outputs_dict = forward_propagation(c_model=c_model, args=args)
    i_loss, b_loss, t_loss = backward_propagation(c_model=c_model, args=args)

    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(
        outputs_dict["true_labels"], outputs_dict["pred_labels"]
    ).ravel()
    train_tn, train_fp, train_fn, train_tp = int(tn), int(fp), int(fn), int(tp)

    train_sensitivity = round(train_tp / (train_tp + train_fn), 2)
    train_specificity = round(train_tn / (train_tn + train_fp), 2)
    train_acc = round(
        (train_tp + train_tn) / (train_tn + train_fp + train_fn + train_tp), 2
    )

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(
        outputs_dict["true_labels"], outputs_dict["pred_labels"], pos_label=1
    )
    train_auc = round(sklearn.metrics.auc(fpr, tpr), 2)

    return {
        "train_loss": t_loss,
        "train_ins_loss": i_loss,
        "train_bag_loss": b_loss,
        "train_sensitivity": train_sensitivity,
        "train_specificity": train_specificity,
        "train_acc": train_acc,
        "train_auc": train_auc,
    }
