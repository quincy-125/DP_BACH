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
import os
import json
import time

from MODEL.model_clam import S_CLAM, M_CLAM
from UTILITY.model_train import train_step
from UTILITY.model_val import val_step
from UTILITY.model_test import test_step
from UTILITY.util import model_save, restore_model, tf_shut_up, str_to_bool


def train_val(
    c_model,
    args,
):
    """_summary_

    Args:
        c_model (_type_): _description_
        args (_type_): _description_
    """
    train_summary_path = os.path.join(args.checkpoints_dir, "summary/train")
    os.makedirs(train_summary_path, exist_ok=True)
    train_summary_writer = tf.summary.create_file_writer(train_summary_path)

    val_summary_path = os.path.join(args.checkpoints_dir, "summary/val")
    os.makedirs(val_summary_path, exist_ok=True)
    val_summary_writer = tf.summary.create_file_writer(val_summary_path)

    train_val_logs = list()
    for epoch in range(args.epochs):
        # Training Step
        start_time = time.time()

        (
            train_loss,
            train_ins_loss,
            train_bag_loss,
            train_tn,
            train_fp,
            train_fn,
            train_tp,
            train_sensitivity,
            train_specificity,
            train_acc,
            train_auc,
        ) = train_step(
            c_model=c_model,
            args=args,
        )

        with train_summary_writer.as_default():
            tf.summary.scalar("Total Loss", float(train_loss), step=epoch)
            tf.summary.scalar("Instance Loss", float(train_ins_loss), step=epoch)
            tf.summary.scalar("Bag Loss", float(train_bag_loss), step=epoch)
            tf.summary.scalar("Accuracy", float(train_acc), step=epoch)
            tf.summary.scalar("AUC", float(train_auc), step=epoch)
            tf.summary.scalar("Sensitivity", float(train_sensitivity), step=epoch)
            tf.summary.scalar("Specificity", float(train_specificity), step=epoch)
            tf.summary.histogram("True Positive", int(train_tp), step=epoch)
            tf.summary.histogram("False Positive", int(train_fp), step=epoch)
            tf.summary.histogram("True Negative", int(train_tn), step=epoch)
            tf.summary.histogram("False Negative", int(train_fn), step=epoch)

        # Validation Step
        (
            val_loss,
            val_ins_loss,
            val_bag_loss,
            val_tn,
            val_fp,
            val_fn,
            val_tp,
            val_sensitivity,
            val_specificity,
            val_acc,
            val_auc,
        ) = val_step(
            c_model=c_model,
            args=args,
        )

        with val_summary_writer.as_default():
            tf.summary.scalar("Total Loss", float(val_loss), step=epoch)
            tf.summary.scalar("Instance Loss", float(val_ins_loss), step=epoch)
            tf.summary.scalar("Bag Loss", float(val_bag_loss), step=epoch)
            tf.summary.scalar("Accuracy", float(val_acc), step=epoch)
            tf.summary.scalar("AUC", float(val_auc), step=epoch)
            tf.summary.scalar("Sensitivity", float(val_sensitivity), step=epoch)
            tf.summary.scalar("Specificity", float(val_specificity), step=epoch)
            tf.summary.histogram("True Positive", int(val_tp), step=epoch)
            tf.summary.histogram("False Positive", int(val_fp), step=epoch)
            tf.summary.histogram("True Negative", int(val_tn), step=epoch)
            tf.summary.histogram("False Negative", int(val_fn), step=epoch)

        epoch_run_time = time.time() - start_time

        # early-stopping
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0.1,
            patience=20,
            mode="min",
            restore_best_weights=True,
        )

        template = (
            "\n Epoch {},  Train Loss: {}, Train Accuracy: {}, Val Loss: {}, Val Accuracy: {}, Epoch Running "
            "Time: {} "
        )
        train_val_log = template.format(
            epoch + 1,
            f"{float(train_loss):.8}",
            f"{float(train_acc):.4%}",
            f"{float(val_loss):.8}",
            f"{float(val_acc):.4%}",
            "--- %s mins ---" % int(epoch_run_time / 60),
        )
        train_val_logs.append(train_val_log)

        print(train_val_log)
    
    train_val_logs_path = os.path.join(args.checkpoints_dir, "logs")
    os.makedirs(train_val_logs_path, exist_ok=True)
    with open(os.path.join(train_val_logs_path, "train_val_log.txt"), "w+") as f:
        for items in train_val_logs:
            f.write("%s\n" %items)


def clam_optimize(
    c_model,
    args,
):
    """_summary_

    Args:
        c_model (_type_): _description_
        args (_type_): _description_
    """
    train_val(
        c_model=c_model,
        args=args,
    )

    model_save(
        c_model=c_model,
        args=args,
    )


def clam_test(
    args,
):
    """_summary_

    Args:
        args (_type_): _description_
    """
    c_trained_model = restore_model(
        args=args,
    )

    test_step(
        c_model=c_trained_model,
        args=args,
    )


def load_model(
    args,
):
    """_summary_

    Args:
        args (_type_): _description_

    Returns:
        _type_: _description_
    """
    s_clam = S_CLAM(
        att_gate=args.att_gate,
        net_size=args.net_size,
        top_k_percent=args.top_k_percent,
        n_class=args.n_class,
        mut_ex=args.mut_ex,
        drop_rate=args.dropout_rate,
        mil_ins=args.mil_ins,
        att_only=args.att_only,
    )

    m_clam = M_CLAM(
        att_gate=args.att_gate,
        net_size=args.net_size,
        top_k_percent=args.top_k_percent,
        n_class=args.n_class,
        mut_ex=args.mut_ex,
        drop_rate=args.dropout_rate,
        mil_ins=args.mil_ins,
        att_only=args.att_only,
    )

    s_clam_model = s_clam
    m_clam_model = m_clam

    if args.m_clam_op:
        c_model = m_clam_model
    else:
        c_model = s_clam_model

    return c_model


def clam(
    args,
):
    """_summary_

    Args:
        args (_type_): _description_
    """
    logging_config_path = os.path.join(args.checkpoints_dir, "config")
    os.makedirs(logging_config_path, exist_ok=True)

    if args.is_training:
        with open(os.path.join(logging_config_path, "train.json"), "w") as f:
            json.dump(dict(args), f)

        c_model = load_model(
            args=args,
        )

        tf_shut_up(no_warn_op=args.no_warn_op)

        if args.m_clam_op:
            b_c_model_index = 1
        else:
            b_c_model_index = 0

        clam_optimize(
            c_model=c_model,
            args=args,
        )
    else:
        with open(os.path.join(logging_config_path, "test.json"), "w") as f:
            json.dump(dict(args), f)

        clam_test(
            args=args,
        )


def clam_main(args,):
    """_summary_

    Args:
        args (_type_): _description_
    """
    if args.multi_gpu:
        gpus = tf.config.experimental.list_logical_devices("GPU")
        if gpus:
            for gpu in gpus:
                with tf.device(gpu.name):
                    clam(
                        args=args,
                    )
    else:
        clam(
            args=args,
        )
