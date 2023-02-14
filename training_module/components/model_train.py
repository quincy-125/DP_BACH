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

from training_module.util import (
    most_frequent,
    get_data_from_tf,
    load_optimizers,
    load_loss_func,
)


## custome optimizing function when NOT applying batch_size
def nb_optimize(
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
    i_optimizer, b_optimizer, a_optimizer = load_optimizers(args=args,)
    i_loss_func, b_loss_func = load_loss_func(args=args,)
    
    gpus = tf.config.list_logical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy(gpus)
    with strategy.scope():
        with tf.GradientTape() as i_tape, tf.GradientTape() as b_tape, tf.GradientTape() as a_tape:
            c_model_dict = c_model.call(img_features, slide_label)

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

    i_grad = i_tape.gradient(I_Loss, c_model.networks()["i_net"].trainable_weights)
    i_optimizer.apply_gradients(zip(i_grad, c_model.networks()["i_net"].trainable_weights))

    b_grad = b_tape.gradient(B_Loss, c_model.networks()["b_net"].trainable_weights)
    b_optimizer.apply_gradients(zip(b_grad, c_model.networks()["b_net"].trainable_weights))

    a_grad = a_tape.gradient(T_Loss, c_model.networks()["a_net"].trainable_weights)
    a_optimizer.apply_gradients(zip(a_grad, c_model.networks()["a_net"].trainable_weights))

    return I_Loss, B_Loss, T_Loss, c_model_dict["predict_slide_label"]


def b_optimize(
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
    i_optimizer, b_optimizer, a_optimizer = load_optimizers(args=args,)
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
            with tf.GradientTape() as i_tape, tf.GradientTape() as b_tape, tf.GradientTape() as a_tape:
                c_model_dict = c_model.call(
                    img_features[step_size : (step_size + args.batch_size)], slide_label
                )

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

            i_grad = i_tape.gradient(Loss_I, c_model.networks()["i_net"].trainable_weights)
            i_optimizer.apply_gradients(zip(i_grad, c_model.networks()["i_net"].trainable_weights))

            b_grad = b_tape.gradient(Loss_B, c_model.networks()["b_net"].trainable_weights)
            b_optimizer.apply_gradients(zip(b_grad, c_model.networks()["b_net"].trainable_weights))

            a_grad = a_tape.gradient(Loss_T, c_model.networks()["a_net"].trainable_weights)
            a_optimizer.apply_gradients(zip(a_grad, c_model.networks()["a_net"].trainable_weights))

        else:
            with tf.GradientTape() as i_tape, tf.GradientTape() as b_tape, tf.GradientTape() as a_tape:
                c_model_dict = c_model.call(
                    img_features[(step_size - n_ins) :], slide_label
                )

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

            i_grad = i_tape.gradient(Loss_I, c_model.networks()["i_net"].trainable_weights)
            i_optimizer.apply_gradients(zip(i_grad, c_model.networks()["i_net"].trainable_weights))

            b_grad = b_tape.gradient(Loss_B, c_model.networks()["b_net"].trainable_weights)
            b_optimizer.apply_gradients(zip(b_grad, c_model.networks()["b_net"].trainable_weights))

            a_grad = a_tape.gradient(Loss_T, c_model.networks()["a_net"].trainable_weights)
            a_optimizer.apply_gradients(zip(a_grad, c_model.networks()["a_net"].trainable_weights))

        Ins_Loss.append(float(Loss_I))
        Bag_Loss.append(float(Loss_B))
        Total_Loss.append(float(Loss_T))

        label_predict.append(c_model_dict["predict_slide_label"])

        step_size += args.batch_size

    I_Loss = statistics.mean(Ins_Loss)
    B_Loss = statistics.mean(Bag_Loss)
    T_Loss = statistics.mean(Total_Loss)

    predict_slide_label = most_frequent(label_predict)

    return I_Loss, B_Loss, T_Loss, predict_slide_label


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

    train_img_uuids = list(pd.read_csv(args.train_data_dir, index_col=False).UUID)
    all_img_uuids = list(os.listdir(args.all_tfrecords_path))

    train_sample_list = [
        os.path.join(args.all_tfrecords_path, img_uuid)
        for img_uuid in all_img_uuids
        if img_uuid.split("_")[-1].split(".tfrecords")[0] in train_img_uuids
    ]

    train_sample_list = random.sample(train_sample_list, len(train_sample_list))
    for i in train_sample_list:
        print("=", end="")
        single_train_data = i
        img_features, slide_label = get_data_from_tf(
            tf_path=single_train_data,
            args=args,
        )
        # shuffle the order of img features list in order to reduce the side effects of randomly drop potential
        # number of patches' feature vectors during training when enable batch training option
        img_features = random.sample(img_features, len(img_features))

        if args.batch_size != 0:
            if args.batch_size < len(img_features):
                I_Loss, B_Loss, T_Loss, predict_slide_label = b_optimize(
                    img_features=img_features,
                    slide_label=slide_label,
                    c_model=c_model,
                    args=args,
                )
            else:
                I_Loss, B_Loss, T_Loss, predict_slide_label = nb_optimize(
                    img_features=img_features,
                    slide_label=slide_label,
                    c_model=c_model,
                    args=args,
                )
        else:
            I_Loss, B_Loss, T_Loss, predict_slide_label = nb_optimize(
                img_features=img_features,
                slide_label=slide_label,
                c_model=c_model,
                args=args,
            )

        loss_total.append(float(T_Loss))
        loss_ins.append(float(I_Loss))
        loss_bag.append(float(B_Loss))

        slide_true_label.append(slide_label)
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
