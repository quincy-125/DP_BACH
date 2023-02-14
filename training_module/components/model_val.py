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
import os
import random
import statistics

from training_module.util import get_data_from_tf, most_frequent, load_loss_func


def nb_val(
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
    i_loss_func, b_loss_func = load_loss_func(args=args,)

    c_model_dict = c_model.call(img_features, slide_label)
    predict_slide_label = c_model_dict["predict_slide_label"]

    ins_loss = list()
    for j in range(len(c_model_dict["ins_logits"])):
        i_loss = i_loss_func(tf.one_hot(c_model_dict["ins_labels"][j], 2), c_model_dict["ins_logits"][j])
        ins_loss.append(i_loss)
    if args.mut_ex:
        I_Loss = (tf.math.add_n(ins_loss) / len(ins_loss)) / args.n_class
    else:
        I_Loss = tf.math.add_n(ins_loss) / len(ins_loss)

    B_Loss = b_loss_func(c_model_dict["Y_true"], c_model_dict["Y_prob"])
    T_Loss = args.c1 * B_Loss + args.c2 * I_Loss

    return I_Loss, B_Loss, T_Loss, predict_slide_label


def b_val(
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
    i_loss_func, b_loss_func = load_loss_func(args=args,)

    step_size = 0

    Ins_Loss = list()
    Bag_Loss = list()
    Total_Loss = list()

    label_predict = list()

    n_ins = args.top_k_percent * args.batch_size
    n_ins = int(n_ins)

    for n_step in range(0, (len(img_features) // args.batch_size + 1)):
        if step_size < (len(img_features) - args.batch_size):
            c_model_dict = c_model.call(
                img_features[step_size : (step_size + args.batch_size)], slide_label
            )
            predict_label = c_model_dict["predict_slide_label"]

            ins_loss = list()
            for j in range(len(c_model_dict["ins_logits"])):
                i_loss = i_loss_func(tf.one_hot(c_model_dict["ins_labels"][j], 2), c_model_dict["ins_logits"][j])
                ins_loss.append(i_loss)
            if args.mut_ex:
                Loss_I = (tf.math.add_n(ins_loss) / len(ins_loss)) / args.n_class
            else:
                Loss_I = tf.math.add_n(ins_loss) / len(ins_loss)

            Loss_B = b_loss_func(c_model_dict["Y_true"], c_model_dict["Y_prob"])
            Loss_T = args.c1 * Loss_B + args.c2 * Loss_I

        else:
            c_model_dict = c_model.call(
                img_features[(step_size - n_ins) :], slide_label
            )
            predict_label = c_model_dict["predict_slide_label"]

            ins_loss = list()
            for j in range(len(c_model_dict["ins_logits"])):
                i_loss = i_loss_func(tf.one_hot(c_model_dict["ins_labels"][j], 2), c_model_dict["ins_logits"][j])
                ins_loss.append(i_loss)
            if args.mut_ex:
                Loss_I = (tf.math.add_n(ins_loss) / len(ins_loss)) / args.n_class
            else:
                Loss_I = tf.math.add_n(ins_loss) / len(ins_loss)

            Loss_B = b_loss_func(c_model_dict["Y_true"], c_model_dict["Y_prob"])

            Loss_T = args.c1 * Loss_B + args.c2 * Loss_I

        Ins_Loss.append(float(Loss_I))
        Bag_Loss.append(float(Loss_B))
        Total_Loss.append(float(Loss_T))

        label_predict.append(predict_label)

        step_size += args.batch_size

    I_Loss = statistics.mean(Ins_Loss)
    B_Loss = statistics.mean(Bag_Loss)
    T_Loss = statistics.mean(Total_Loss)

    predict_slide_label = most_frequent(label_predict)

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

    val_img_uuids = list(pd.read_csv(args.val_data_dir, index_col=False).UUID)
    all_img_uuids = list(os.listdir(args.all_tfrecords_path))

    val_sample_list = [
        os.path.join(args.all_tfrecords_path, img_uuid)
        for img_uuid in all_img_uuids
        if img_uuid.split("_")[-1].split(".tfrecords")[0] in val_img_uuids
    ]

    for i in val_sample_list:
        print("=", end="")
        single_val_data = i
        img_features, slide_label = get_data_from_tf(
            tf_path=single_val_data,
            args=args,
        )
        img_features = random.sample(
            img_features, len(img_features)
        )  # follow the training loop, see details there

        if args.batch_size != 0:
            if args.batch_size < len(img_features):
                I_Loss, B_Loss, T_Loss, predict_slide_label = b_val(
                    img_features=img_features,
                    slide_label=slide_label,
                    c_model=c_model,
                    args=args,
                )
            else:
                I_Loss, B_Loss, T_Loss, predict_slide_label = nb_val(
                    img_features=img_features,
                    slide_label=slide_label,
                    c_model=c_model,
                    args=args,
                )
        else:
            I_Loss, B_Loss, T_Loss, predict_slide_label = nb_val(
                img_features=img_features,
                slide_label=slide_label,
                c_model=c_model,
                args=args,
            )

        loss_t.append(float(T_Loss))
        loss_i.append(float(I_Loss))
        loss_b.append(float(B_Loss))

        slide_true_label.append(slide_label)
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