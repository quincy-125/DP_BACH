# Copyright 2022 Mayo Clinic. All Rights Reserved.
#
# Author: Quincy Gu (M216613)
# Affliation: Division of Computational Pathology and Artificial Intelligence,
# Department of Laboratory Medicine and Pathology, Mayo Clinic College of Medicine and Science
# Email: Gu.Qiangqiang@mayo.edu
# Version: 1.0.1
# Created on: 11/27/2022 4:40 pm CST
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


import os
import argparse
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold

import sys

sys_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(sys_dir))

from components.utils import *

configure_logging(script_name="cross_val_split")


def binary_bach_data(data_path, neg_labels, pos_labels):
    """_summary_

    Args:
        data_path (_type_): _description_
        neg_labels (_type_): _description_
        pos_labels (_type_): _description_

    Returns:
        _type_: _description_
    """
    logging.info("reorganize BACH dataset for binary classification")

    neg_slides = list()
    pos_slides = list()

    for i in neg_labels:
        for neg in tf.io.gfile.listdir(os.path.join(data_path, i)):
            neg_slides.append(neg)

    for j in pos_labels:
        for pos in tf.io.gfile.listdir(os.path.join(data_path, j)):
            pos_slides.append(pos)

    return neg_slides, pos_slides


def select_test_data(neg_slides, pos_slides, test_ratio=0.2):
    """_summary_

    Args:
        neg_slides (_type_): _description_
        pos_slides (_type_): _description_
        test_ratio (float, optional): _description_. Defaults to 0.2.

    Returns:
        _type_: _description_
    """
    import random

    neg_test_slides = random.choices(neg_slides, k=int(len(neg_slides) * test_ratio))

    pos_test_slides = random.choices(pos_slides, k=int(len(pos_slides) * test_ratio))

    return neg_test_slides, pos_test_slides


def cross_val_data(
    neg_slides,
    pos_slides,
    kf_csv_path,
    test_ratio=0.0,
    n_folds=5,
    kf_shuffle=False,
    kf_rs=None,
):
    """_summary_

    Args:
        neg_slides (_type_): _description_
        pos_slides (_type_): _description_
        kf_csv_path (_type_): _description_
        test_ratio (float, optional): _description_. Defaults to 0.0.
        n_folds (int, optional): _description_. Defaults to 5.
        kf_shuffle (bool, optional): _description_. Defaults to False.
        kf_rs (_type_, optional): _description_. Defaults to None.
    """
    os.makedirs(kf_csv_path, exist_ok=True)

    if test_ratio != 0.0:
        neg_test_slides, pos_test_slides = select_test_data(
            neg_slides=neg_slides, pos_slides=pos_slides, test_ratio=test_ratio
        )
        test_df = pd.DataFrame({"UUID": neg_test_slides + pos_test_slides})
        ## Write K-Fold Cross Validation Data Split into CSV File
        test_kf_csv_path = os.path.join(kf_csv_path, "test")
        os.makedirs(test_kf_csv_path, exist_ok=True)
        test_df.to_csv(
            "{}/bach_ratio_{}_p_{}_test.csv".format(test_kf_csv_path, str(test_ratio).split(".")[0], str(test_ratio).split(".")[-1]),
            index=False,
        )

        ## return slides uuids for training and validation from the negative and positive class samples
        neg_slides = list(set(neg_slides) - set(neg_test_slides))
        pos_slides = list(set(pos_slides) - set(pos_test_slides))

    ## initiate k-fold cross validation, check sklearn KFold documentation via
    ## https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html for
    ## more information in details
    kf_cv = KFold(n_splits=n_folds, shuffle=kf_shuffle, random_state=kf_rs)

    ## K-Fold Cross Validation Data Split in Negative Class Samples
    neg_train_index = list()
    neg_val_index = list()
    for nt_idx, nv_idx in kf_cv.split(neg_slides):
        neg_train_index.append(nt_idx)
        neg_val_index.append(nv_idx)

    neg_train_slides = list()
    for fold_nt_idx in neg_train_index:
        neg_train_slides.append([neg_slides[i] for i in fold_nt_idx])

    neg_val_slides = list()
    for fold_nv_idx in neg_val_index:
        neg_val_slides.append([neg_slides[i] for i in fold_nv_idx])

    ## K-Fold Cross Validation Data Split in Postive Class Samples
    pos_train_index = list()
    pos_val_index = list()
    for pt_idx, pv_idx in kf_cv.split(pos_slides):
        pos_train_index.append(pt_idx)
        pos_val_index.append(pv_idx)

    pos_train_slides = list()
    for fold_pt_idx in pos_train_index:
        pos_train_slides.append([pos_slides[i] for i in fold_pt_idx])

    pos_val_slides = list()
    for fold_pv_idx in pos_val_index:
        pos_val_slides.append([pos_slides[i] for i in fold_pv_idx])

    ## Combine K-Fold Cross Validation Data Split in Negative and Positive Class Samples
    for f in range(n_folds):
        train_fold_slides = neg_train_slides[f] + pos_train_slides[f]
        val_fold_slides = neg_val_slides[f] + pos_val_slides[f]
        train_fold_df = pd.DataFrame({"UUID": train_fold_slides})
        val_fold_df = pd.DataFrame({"UUID": val_fold_slides})

        ## Write K-Fold Cross Validation Data Split into CSV File
        fold_kf_csv_path = os.path.join(kf_csv_path, "fold_{}".format(f+1))
        os.makedirs(fold_kf_csv_path, exist_ok=True)

        train_fold_df.to_csv(
            "{}/bach_fold_{}_train.csv".format(fold_kf_csv_path, f + 1),
            index=False
        )
        val_fold_df.to_csv(
            "{}/bach_fold_{}_val.csv".format(fold_kf_csv_path, f + 1), 
            index=False
        )


def run_kf_cross_val(
    data_path,
    neg_labels,
    pos_labels,
    kf_csv_path,
    test_ratio=0.0,
    n_folds=5,
    kf_shuffle=False,
    kf_rs=None,
):
    """_summary_

    Args:
        data_path (_type_): _description_
        neg_labels (_type_): _description_
        pos_labels (_type_): _description_
        kf_csv_path (_type_): _description_
        test_ratio (float, optional): _description_. Defaults to 0.0.
        n_folds (int, optional): _description_. Defaults to 5.
        kf_shuffle (bool, optional): _description_. Defaults to False.
        kf_rs (_type_, optional): _description_. Defaults to None.
    """
    neg_slides, pos_slides = binary_bach_data(
        data_path=data_path, neg_labels=neg_labels, pos_labels=pos_labels
    )

    logging.info(
        "Starting to Execute {}-Fold Cross Validation Data Split".format(n_folds)
    )
    cross_val_data(
        neg_slides=neg_slides,
        pos_slides=pos_slides,
        kf_csv_path=kf_csv_path,
        test_ratio=test_ratio,
        n_folds=n_folds,
        kf_shuffle=kf_shuffle,
        kf_rs=kf_rs,
    )
    logging.info(
        "Successfully Completed the Execution of {}-Fold Cross Validation Data Split".format(
            n_folds
        )
    )
