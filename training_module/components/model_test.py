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


import pandas as pd
import sklearn
import os
import time

from training_module.util import (
    get_data_from_tf,
    s_clam_call,
    most_frequent,
    m_clam_call,
)


def m_test_per_sample(
    c_model,
    img_features,
    slide_label,
    args,
):
    """_summary_

    Args:
        c_model (_type_): _description_
        img_features (_type_): _description_
        slide_label (_type_): _description_
        args (_type_): _description_

    Returns:
        _type_: _description_
    """
    slide_pred_per_sample = list()

    for i in range(args.test_steps):
        if args.m_clam_op:
            predict_label = m_clam_call(
                att_net=c_model[0],
                ins_net=c_model[1],
                bag_net=c_model[2],
                img_features=img_features,
                slide_label=slide_label,
                args=args,
            )["predict_label"]
        else:
            predict_label = s_clam_call(
                att_net=c_model[0],
                ins_net=c_model[1],
                bag_net=c_model[2],
                img_features=img_features,
                slide_label=slide_label,
                args=args,
            )["predict_label"]

        slide_pred_per_sample.append(predict_label)
        predict_slide_label = most_frequent(slide_pred_per_sample)

        return predict_slide_label


def test_step(
    c_model,
    args,
):
    """_summary_

    Args:
        c_model (_type_): _description_
        args (_type_): _description_
    """
    start_time = time.time()

    slide_true_label = list()
    slide_predict_label = list()
    sample_names = list()

    test_img_uuids = list(pd.read_csv(args.test_data_dir, index_col=False).UUID)
    all_img_uuids = list(os.listdir(args.all_tfrecords_path))

    test_sample_list = [
        os.path.join(args.all_tfrecords_path, img_uuid)
        for img_uuid in all_img_uuids
        if img_uuid.split("_")[-1].split(".tfrecords")[0] in test_img_uuids
    ]

    for i in test_sample_list:
        print(">", end="")
        single_test_data = i
        img_features, slide_label = get_data_from_tf(
            tf_path=single_test_data,
            args=args,
        )

        predict_slide_label = m_test_per_sample(
            c_model=c_model,
            img_features=img_features,
            slide_label=slide_label,
            args=args,
        )

        slide_true_label.append(slide_label)
        slide_predict_label.append(predict_slide_label)
        sample_names.append(i)

        test_results = pd.DataFrame(
            list(zip(sample_names, slide_true_label, slide_predict_label)),
            columns=["Sample Names", "Slide True Label", "Slide Predict Label"],
        )
        test_results.to_csv(
            os.path.join(args.test_result_dir, args.test_result_file_name),
            sep="\t",
            index=False,
        )

    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(
        slide_true_label, slide_predict_label
    ).ravel()
    test_tn = int(tn)
    test_fp = int(fp)
    test_fn = int(fn)
    test_tp = int(tp)

    test_sensitivity = round(test_tp / (test_tp + test_fn), 2)
    test_specificity = round(test_tn / (test_tn + test_fp), 2)
    test_acc = round((test_tp + test_tn) / (test_tn + test_fp + test_fn + test_tp), 2)

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(
        slide_true_label, slide_predict_label, pos_label=1
    )
    test_auc = round(sklearn.metrics.auc(fpr, tpr), 2)

    test_run_time = time.time() - start_time

    template = "\n Test Accuracy: {}, Test AUC: {}, Test Sensitivity: {}, Test Specificity: {}, Test Running Time: {}"
    print(
        template.format(
            f"{float(test_acc):.4%}",
            f"{float(test_auc):.4%}",
            f"{float(test_sensitivity):.4%}",
            f"{float(test_specificity):.4%}",
            "--- %s mins ---" % int(test_run_time / 60),
        )
    )
