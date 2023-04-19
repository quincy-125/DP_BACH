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
    most_frequent,
    get_data_from_tf,
    load_sample_dataset,
    load_optimizers,
    load_loss_func,
)

os.environ["HYDRA_FULL_ERROR"] = "1"


def optimize(
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
                c_model_dict = c_model.call(img_features, slide_label)

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
        with tf.GradientTape() as i_tape, tf.GradientTape() as b_tape, tf.GradientTape() as a_tape:
            c_model_dict = c_model.call(img_features, slide_label)

            i_loss = i_loss_func(
                c_model_dict["ins_labels"],
                c_model_dict["ins_logits"],
            )
            I_Loss = tf.math.reduce_mean(i_loss)
            if args.mut_ex:
                I_Loss = I_Loss / args.n_class
            B_Loss = b_loss_func(c_model_dict["Y_true"], c_model_dict["Y_prob"])
            T_Loss = args.c1 * B_Loss + args.c2 * I_Loss

    i_grad = i_tape.gradient(I_Loss, c_model.networks()["i_net"].trainable_weights)
    i_optimizer.apply_gradients(
        zip(i_grad, c_model.networks()["i_net"].trainable_weights)
    )

    b_grad = b_tape.gradient(B_Loss, c_model.networks()["b_net"].trainable_weights)
    b_optimizer.apply_gradients(
        zip(b_grad, c_model.networks()["b_net"].trainable_weights)
    )

    a_grad = a_tape.gradient(T_Loss, c_model.networks()["a_net"].trainable_weights)
    a_optimizer.apply_gradients(
        zip(a_grad, c_model.networks()["a_net"].trainable_weights)
    )

    return I_Loss, B_Loss, T_Loss, c_model_dict["predict_slide_label"]


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
    loss_total = list()
    loss_ins = list()
    loss_bag = list()

    slide_true_label = list()
    slide_predict_label = list()

    train_sample_dataset = load_sample_dataset(args=args, sample_name="train")
    features, labels = (
        train_sample_dataset["image_features"],
        train_sample_dataset["slide_labels"],
    )
    for i in range(len(labels)):
        print("=", end="")
        I_Loss, B_Loss, T_Loss, predict_slide_label = optimize(
            img_features=features[i],
            slide_label=labels[i],
            c_model=c_model,
            args=args,
        )

        loss_total.append(float(T_Loss))
        loss_ins.append(float(I_Loss))
        loss_bag.append(float(B_Loss))

        slide_true_label.append(labels[i])
        slide_predict_label.append(predict_slide_label)

    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(
        slide_true_label, slide_predict_label
    ).ravel()
    train_tn = int(tn)
    train_fp = int(fp)
    train_fn = int(fn)
    train_tp = int(tp)

    train_sensitivity = round(train_tp / (train_tp + train_fn), 2)
    train_specificity = round(train_tn / (train_tn + train_fp), 2)
    train_acc = round(
        (train_tp + train_tn) / (train_tn + train_fp + train_fn + train_tp), 2
    )

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(
        slide_true_label, slide_predict_label, pos_label=1
    )
    train_auc = round(sklearn.metrics.auc(fpr, tpr), 2)

    train_loss = statistics.mean(loss_total)
    train_ins_loss = statistics.mean(loss_ins)
    train_bag_loss = statistics.mean(loss_bag)

    return {
        "train_loss": train_loss,
        "train_ins_loss": train_ins_loss,
        "train_bag_loss": train_bag_loss,
        "train_tn": train_tn,
        "train_fp": train_fp,
        "train_fn": train_fn,
        "train_tp": train_tp,
        "train_sensitivity": train_sensitivity,
        "train_specificity": train_specificity,
        "train_acc": train_acc,
        "train_auc": train_auc,
    }
