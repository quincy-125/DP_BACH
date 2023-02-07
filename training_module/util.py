# Copyright 2022 Mayo Clinic. All Rights Reserved.
#
# Author: Quincy Gu (M216613)
# Affliation: Division of Computational Pathology and Artificial Intelligence,
# Department of Laboratory Medicine and Pathology, Mayo Clinic College of Medicine and Science
# Email: Gu.Qiangqiang@mayo.edu
# Version: 1.0.1
# Created on: 11/27/2022 6:35 pm CST
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
import random
import shutil
import subprocess

import numpy as np
import tensorflow as tf

from pathlib import Path

import sys
import logging

sys_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(sys_dir))


def resolve_gcs_uri_to_local_uri(gcs_file_uri: str) -> str:
    """
    Use an existing gcsfuse mountpoint, or use gcsfuse to mount a bucket
    locally if not mountpoint exists, to get a local filepath for a file stored
    in a GCS bucket.

    Args:
        gcs_file_uri (str): A "gs:/..." URI for a file stored in a GCS bucket.

    Returns:
        str: A string indicating a local filepath where gcs_file_uri can be
        found.
    """
    if not gcs_file_uri.startswith("gs:/"):
        raise Exception('Please use a "gs:/" GCS path.')

    # Pathlib isn't wholly apposite to use for URLs (specifically, it changes
    # 'gs://...' to 'gs:/...'), but it's useful in this case:
    gcs_path = Path(gcs_file_uri)
    bucket_name = gcs_path.parts[1]
    file_remaining_path = Path(*gcs_path.parts[2:])
    logging.info("Checking for existing mountpoint for %s...", bucket_name)
    existing_mountpoint_check = subprocess.run(
        f'findmnt --noheadings --output TARGET --source "{bucket_name}"',
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    existing_mountpoint_check_stdout = existing_mountpoint_check.stdout.decode().strip()

    if existing_mountpoint_check.returncode == 127:
        raise Exception(
            f'Error when running findmnt: "{existing_mountpoint_check_stdout}"'
        )

    if (
        existing_mountpoint_check.returncode == 0
        and existing_mountpoint_check_stdout != ""
    ):
        # An existing mountpoint for the bucket exists; we will reuse it here,
        # if it contains the file we're looking for (it's possible that the
        # bucket is mounted, but to a different subdirectory within the bucket
        # than we need).
        for mountpoint in existing_mountpoint_check_stdout.split("\n"):
            logging.info(
                'Checking for file "%s" at "%s"...', file_remaining_path, mountpoint
            )
            if Path(mountpoint, file_remaining_path).exists():
                found_file_path = os.path.join(mountpoint, file_remaining_path)
                logging.info('Found file at "%s"', found_file_path)
                return found_file_path

    # No existing mountpoint exists, so we will mount the bucket with gcsfuse:
    import tempfile

    tmp_directory = tempfile.mkdtemp()
    logging.info('Creating new mountpoint at "%s"...', tmp_directory)
    mount_command = subprocess.run(
        # Unexpectedly, using `f"gcsfuse --implicit-dirs --only-dir {specific_bucket_directory} {bucket_name} {tmp_directory}"`
        # resulted in a mountpoint that was *unusably slow* when trying to,
        # e.g., using `.get_thumbnail()`. Thus, here, we are not using
        # `--only-dir.`
        f"gcsfuse --implicit-dirs {bucket_name} {tmp_directory}",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    mount_command_stdout = mount_command.stdout.decode().strip()
    if "File system has been successfully mounted" not in mount_command_stdout:
        raise Exception(
            f'Error when running gcsfuse: "{mount_command_stdout}". Exit code was {existing_mountpoint_check.returncode}.'
        )

    return os.path.join(tmp_directory, file_remaining_path)


def connect_path_arg_gcs(args):
    """_summary_

    Args:
        args (_type_): _description_
    """
    mount_base_path = resolve_gcs_uri_to_local_uri(gcs_file_uri=args.gcs_file_uri)

    args.all_tfrecords_path = os.path.join(mount_base_path, args.all_tfrecords_path)
    args.train_data_dir = os.path.join(mount_base_path, args.train_data_dir)
    args.val_data_dir=os.path.join(mount_base_path, args.val_data_dir)
    args.checkpoints_dir = os.path.join(mount_base_path, args.checkpoints_dir)
    args.stdout_path = os.path.join(mount_base_path, args.stdout_path)


def configure_logging(
    script_name,
    args,
):
    """_summary_

    Args:
        script_name (_type_): _description_
        args (_type_): _description_

    Returns:
        _type_: _description_
    """
    log_format = "%(asctime)s: %(name)s: %(levelname)s: %(message)s"

    os.makedirs("../logs", exist_ok=True)
    with open(args.stdout_path, "w") as f:
        pass

    logging.basicConfig(
        level=logging.DEBUG,
        format=log_format,
        filename=args.stdout_path,
        filemode="w",
    )

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter(log_format))

    logging.getLogger(script_name).addHandler(console)

    return logging.getLogger(script_name)


def get_data_from_tf(
    tf_path,
    args,
):
    """_summary_

    Args:
        tf_path (_type_): _description_
        args (_type_): _description_

    Returns:
        _type_: _description_
    """
    feature = {
        "height": tf.io.FixedLenFeature([], tf.int64),
        "width": tf.io.FixedLenFeature([], tf.int64),
        "depth": tf.io.FixedLenFeature([], tf.int64),
        "label": tf.io.FixedLenFeature([], tf.int64),
        "image/format": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string),
        "image/encoded": tf.io.FixedLenFeature([], tf.string),
        "image_feature": tf.io.FixedLenFeature([], tf.string),
    }

    tfrecord_dataset = tf.data.TFRecordDataset(tf_path)

    def _parse_image_function(key):
        return tf.io.parse_single_example(key, feature)

    CLAM_dataset = tfrecord_dataset.map(_parse_image_function)

    image_features = list()

    for tfrecord_value in CLAM_dataset:
        img_feature = tf.io.parse_tensor(tfrecord_value["image_feature"], "float32")

        if args.imf_norm_op:
            img_feature = tf.math.l2_normalize(img_feature)

        slide_labels = tfrecord_value["label"]
        slide_label = int(slide_labels)

        image_features.append(img_feature)

    return image_features, slide_label


def most_frequent(list):
    """_summary_

    Args:
        list (_type_): _description_

    Returns:
        _type_: _description_
    """
    mf = max(set(list), key=list.count)
    return mf


def tf_shut_up(no_warn_op=False):
    """_summary_

    Args:
        no_warn_op (bool, optional): _description_. Defaults to False.
    """
    if no_warn_op:
        tf.get_logger().setLevel("ERROR")
    else:
        print(
            "Are you sure you want to receive the annoying TensorFlow Warning Messages?",
            "\n",
            "If not, check the value of your input prameter for this function and re-run it.",
        )


def optimizer_func_options(
    args,
):
    """_summary_

    Args:
        args (_type_): _description_

    Returns:
        _type_: _description_
    """
    optimizer_func_dic = {
        "Adam": tf.keras.optimizers.Adam,
        "Adadelta": tf.keras.optimizers.Adadelta,
        "Adagrad": tf.keras.optimizers.Adagrad,
        "Adamax": tf.keras.optimizers.Adamax,
        "Ftrl": tf.keras.optimizers.Ftrl,
        "Nadam": tf.keras.optimizers.Nadam,
        "RMSprop": tf.keras.optimizers.RMSprop,
        "SGD": tf.keras.optimizers.SGD,
    }

    return optimizer_func_dic


def loss_func_options():
    """_summary_

    Returns:
        _type_: _description_
    """
    loss_func_dic = {
        "binary_crossentropy": tf.keras.losses.binary_crossentropy,
        "hinge": tf.keras.losses.hinge,
        "categorical_crossentropy": tf.keras.losses.categorical_crossentropy,
        "categorical_hinge": tf.keras.losses.categorical_hinge,
        "cosine_similarity": tf.keras.losses.cosine_similarity,
        "huber": tf.keras.losses.huber,
        "log_cosh": tf.keras.losses.log_cosh,
        "poisson": tf.keras.losses.poisson,
        "squared_hinge": tf.keras.losses.squared_hinge,
    }

    return loss_func_dic


def load_optimizers(
    args,
):
    """_summary_

    Args:
        args (_type_): _description_

    Returns:
        _type_: _description_
    """
    i_optimizer_func = optimizer_func_options(
        args=args,
    )[args.i_optimizer_name]
    b_optimizer_func = optimizer_func_options(
        args=args,
    )[args.b_optimizer_name]
    c_optimizer_func = optimizer_func_options(
        args=args,
    )[args.a_optimizer_name]

    i_optimizer = i_optimizer_func(learning_rate=args.i_learn_rate)
    b_optimizer = b_optimizer_func(learning_rate=args.b_learn_rate)
    c_optimizer = c_optimizer_func(learning_rate=args.a_learn_rate)

    return i_optimizer, b_optimizer, c_optimizer


def load_loss_func(
    args,
):
    """_summary_

    Args:
        args (_type_): _description_

    Returns:
        _type_: _description_
    """
    i_loss_func = loss_func_options()[args.i_loss_name]
    b_loss_func = loss_func_options()[args.b_loss_name]

    return i_loss_func, b_loss_func


def str_to_bool():
    """_summary_

    Returns:
        _type_: _description_
    """
    str_bool_dic = {"True": True, "False": False}

    return str_bool_dic


def dataset_shuffle(dataset, path, percent):
    """
    Input Arg:
        dataset -> path where all tfrecord data stored
        path -> path where you want to save training, testing, and validation data folder
    """

    # return training, validation, and testing path name
    train = path + "/train"
    valid = path + "/valid"
    test = path + "/test"

    # create training, validation, and testing directory only if it is not existed
    os.makedirs(os.path.join(path, "train"), exist_ok=True)

    os.makedirs(os.path.join(path, "valid"), exist_ok=True)

    os.makedirs(os.path.join(path, "test"), exist_ok=True)

    total_num_data = len(os.listdir(dataset))

    # only shuffle the data when train, validation, and test directory are all empty
    if (
        len(os.listdir(train))
        == 0 & len(os.listdir(valid))
        == 0 & len(os.listdir(test))
        == 0
    ):
        train_names = random.sample(
            os.listdir(dataset), int(total_num_data * percent[0])
        )
        for i in train_names:
            train_srcpath = os.path.join(dataset, i)
            shutil.copy(train_srcpath, train)

        valid_names = random.sample(
            list(set(os.listdir(dataset)) - set(os.listdir(train))),
            int(total_num_data * percent[1]),
        )
        for j in valid_names:
            valid_srcpath = os.path.join(dataset, j)
            shutil.copy(valid_srcpath, valid)

        test_names = list(
            set(os.listdir(dataset)) - set(os.listdir(train)) - set(os.listdir(valid))
        )
        for k in test_names:
            test_srcpath = os.path.join(dataset, k)
            shutil.copy(test_srcpath, test)


def ng_att_call(ng_att_net, img_features):
    """_summary_

    Args:
        ng_att_net (_type_): _description_
        img_features (_type_): _description_

    Returns:
        _type_: _description_
    """
    h = list()
    A = list()

    for i in img_features:
        c_imf = ng_att_net[0](i)
        h.append(c_imf)

    for j in h:
        a = ng_att_net[1](j)
        A.append(a)
    return h, A


def g_att_call(g_att_net, img_features):
    """_summary_

    Args:
        g_att_net (_type_): _description_
        img_features (_type_): _description_

    Returns:
        _type_: _description_
    """
    h = list()
    A = list()

    for i in img_features:
        c_imf = g_att_net[0](i)
        h.append(c_imf)

    for j in h:
        att_v_output = g_att_net[1](j)
        att_u_output = g_att_net[2](j)
        att_input = tf.math.multiply(att_v_output, att_u_output)
        a = g_att_net[3](att_input)
        A.append(a)

    return h, A


def generate_pos_labels(n_pos_sample):
    """_summary_

    Args:
        n_pos_sample (_type_): _description_

    Returns:
        _type_: _description_
    """
    return tf.fill(
        dims=[
            n_pos_sample,
        ],
        value=1,
    )


def generate_neg_labels(n_neg_sample):
    """_summary_

    Args:
        n_neg_sample (_type_): _description_

    Returns:
        _type_: _description_
    """
    return tf.fill(
        dims=[
            n_neg_sample,
        ],
        value=0,
    )


def ins_in_call(
    ins_classifier,
    h,
    A_I,
    args,
):
    """_summary_

    Args:
        ins_classifier (_type_): _description_
        h (_type_): _description_
        A_I (_type_): _description_
        args (_type_): _description_

    Returns:
        _type_: _description_
    """
    n_ins = args.top_k_percent * len(h)
    n_ins = int(n_ins)

    pos_label = generate_pos_labels(n_pos_sample=n_ins)
    neg_label = generate_neg_labels(n_neg_sample=n_ins)
    ins_label_in = tf.concat(values=[pos_label, neg_label], axis=0)

    A_I = tf.reshape(tf.convert_to_tensor(A_I), (1, len(A_I)))

    top_pos_ids = tf.math.top_k(A_I, n_ins)[1][-1]
    pos_index = list()
    for i in top_pos_ids:
        pos_index.append(i)

    pos_index = tf.convert_to_tensor(pos_index)
    top_pos = list()
    for i in pos_index:
        top_pos.append(h[i])

    top_neg_ids = tf.math.top_k(-A_I, n_ins)[1][-1]
    neg_index = list()
    for i in top_neg_ids:
        neg_index.append(i)

    neg_index = tf.convert_to_tensor(neg_index)
    top_neg = list()
    for i in neg_index:
        top_neg.append(h[i])

    ins_in = tf.concat(values=[top_pos, top_neg], axis=0)
    logits_unnorm_in = list()
    logits_in = list()

    for i in range(args.n_class * n_ins):
        ins_score_unnorm_in = ins_classifier(ins_in[i])
        logit_in = tf.math.softmax(ins_score_unnorm_in)
        logits_unnorm_in.append(ins_score_unnorm_in)
        logits_in.append(logit_in)

    return ins_label_in, logits_unnorm_in, logits_in


def ins_out_call(
    ins_classifier,
    h,
    A_O,
    args,
):
    """_summary_

    Args:
        ins_classifier (_type_): _description_
        h (_type_): _description_
        A_O (_type_): _description_
        args (_type_): _description_

    Returns:
        _type_: _description_
    """
    n_ins = args.top_k_percent * len(h)
    n_ins = int(n_ins)

    # get compressed 512-dimensional instance-level feature vectors for following use, denoted by h
    A_O = tf.reshape(tf.convert_to_tensor(A_O), (1, len(A_O)))

    top_pos_ids = tf.math.top_k(A_O, n_ins)[1][-1]
    pos_index = list()
    for i in top_pos_ids:
        pos_index.append(i)

    pos_index = tf.convert_to_tensor(pos_index)
    top_pos = list()
    for i in pos_index:
        top_pos.append(h[i])

    # mutually-exclusive -> top k instances w/ highest attention scores ==> false pos = neg
    pos_ins_labels_out = generate_neg_labels(n_neg_sample=n_ins)
    ins_label_out = pos_ins_labels_out

    logits_unnorm_out = list()
    logits_out = list()

    for i in range(n_ins):
        ins_score_unnorm_out = ins_classifier(top_pos[i])
        logit_out = tf.math.softmax(ins_score_unnorm_out)
        logits_unnorm_out.append(ins_score_unnorm_out)
        logits_out.append(logit_out)

    return ins_label_out, logits_unnorm_out, logits_out


def ins_call(
    m_ins_classifier,
    bag_label,
    h,
    A,
    args,
):
    """_summary_

    Args:
        m_ins_classifier (_type_): _description_
        bag_label (_type_): _description_
        h (_type_): _description_
        A (_type_): _description_
        args (_type_): _description_

    Returns:
        _type_: _description_
    """
    for i in range(args.n_class):
        ins_classifier = m_ins_classifier[i]
        if i == bag_label:
            A_I = list()
            for j in range(len(A)):
                a_i = A[j][0][i]
                A_I.append(a_i)
            ins_label_in, logits_unnorm_in, logits_in = ins_in_call(
                ins_classifier=ins_classifier,
                h=h,
                A_I=A_I,
                top_k_percent=args.top_k_percent,
                n_class=args.n_class,
            )
        else:
            if args.mut_ex:
                A_O = list()
                for j in range(len(A)):
                    a_o = A[j][0][i]
                    A_O.append(a_o)
                ins_label_out, logits_unnorm_out, logits_out = ins_out_call(
                    ins_classifier=ins_classifier,
                    h=h,
                    A_O=A_O,
                    top_k_percent=args.top_k_percent,
                )
            else:
                continue

    if args.mut_ex:
        ins_labels = tf.concat(values=[ins_label_in, ins_label_out], axis=0)
        ins_logits_unnorm = logits_unnorm_in + logits_unnorm_out
        ins_logits = logits_in + logits_out
    else:
        ins_labels = ins_label_in
        ins_logits_unnorm = logits_unnorm_in
        ins_logits = logits_in

    return ins_labels, ins_logits_unnorm, ins_logits


def s_bag_h_slide(A, h):
    """_summary_

    Args:
        A (_type_): _description_
        h (_type_): _description_

    Returns:
        _type_: _description_
    """
    # compute the slide-level representation aggregated per the attention score distribution for the mth class
    SAR = list()
    for i in range(len(A)):
        sar = tf.linalg.matmul(tf.transpose(A[i]), h[i])  # shape be (2,512)
        SAR.append(sar)

    slide_agg_rep = tf.math.add_n(SAR)  # return h_[slide,m], shape be (2,512)

    return slide_agg_rep


def s_bag_call(
    bag_classifier,
    bag_label,
    A,
    h,
    args,
):
    """_summary_

    Args:
        bag_classifier (_type_): _description_
        bag_label (_type_): _description_
        A (_type_): _description_
        h (_type_): _description_
        args (_type_): _description_

    Returns:
        _type_: _description_
    """
    slide_agg_rep = s_bag_h_slide(A=A, h=h)

    slide_score_unnorm = bag_classifier(slide_agg_rep)
    slide_score_unnorm = tf.reshape(slide_score_unnorm, (1, args.n_class))

    Y_hat = tf.math.top_k(slide_score_unnorm, 1)[1][-1]

    Y_prob = tf.math.softmax(
        tf.reshape(slide_score_unnorm, (1, args.n_class))
    )  # shape be (1,2), predictions for each of the classes

    predict_slide_label = np.argmax(Y_prob.numpy())

    Y_true = tf.one_hot([bag_label], 2)

    return slide_score_unnorm, Y_hat, Y_prob, predict_slide_label, Y_true


def m_bag_h_slide(
    A,
    h,
    args,
):
    """_summary_

    Args:
        A (_type_): _description_
        h (_type_): _description_
        args (_type_): _description_

    Returns:
        _type_: _description_
    """
    SAR = list()
    for i in range(len(A)):
        sar = tf.linalg.matmul(tf.transpose(A[i]), h[i])  # shape be (2,512)
        SAR.append(sar)

    SAR_Branch = list()
    for i in range(args.n_class):
        sar_branch = list()
        for j in range(len(SAR)):
            sar_c = tf.reshape(SAR[j][i], (1, args.dim_compress_features))
            sar_branch.append(sar_c)
        SAR_Branch.append(sar_branch)

    slide_agg_rep = list()
    for k in range(args.n_class):
        slide_agg_rep.append(tf.math.add_n(SAR_Branch[k]))

    return slide_agg_rep


def m_bag_call(
    m_bag_classifier,
    bag_label,
    A,
    h,
    args,
):
    """_summary_

    Args:
        m_bag_classifier (_type_): _description_
        bag_label (_type_): _description_
        A (_type_): _description_
        h (_type_): _description_
        args (_type_): _description_

    Returns:
        _type_: _description_
    """
    slide_agg_rep = m_bag_h_slide(
        A=A, h=h, dim_compress_features=args.dim_compress_features, n_class=args.n_class
    )

    ssus = list()
    # return s_[slide,m] (slide-level prediction scores)
    for i in range(args.n_class):
        bag_classifier = m_bag_classifier[i]
        ssu = bag_classifier(slide_agg_rep[i])
        ssus.append(ssu[0][0])

    slide_score_unnorm = tf.convert_to_tensor(ssus)
    slide_score_unnorm = tf.reshape(slide_score_unnorm, (1, args.n_class))

    Y_hat = tf.math.top_k(slide_score_unnorm, 1)[1][-1]
    Y_prob = tf.math.softmax(slide_score_unnorm)
    predict_slide_label = np.argmax(Y_prob.numpy())

    Y_true = tf.one_hot([bag_label], 2)

    return slide_score_unnorm, Y_hat, Y_prob, predict_slide_label, Y_true


def s_clam_call(
    att_net,
    ins_net,
    bag_net,
    img_features,
    slide_label,
    args,
):
    """_summary_

    Args:
        att_net (_type_): _description_
        ins_net (_type_): _description_
        bag_net (_type_): _description_
        img_features (_type_): _description_
        slide_label (_type_): _description_
        args (_type_): _description_

    Returns:
        _type_: _description_
    """
    if args.att_gate:
        h, A = g_att_call(g_att_net=att_net, img_features=img_features)
    else:
        h, A = ng_att_call(ng_att_net=att_net, img_features=img_features)
    att_score = A  # output from attention network
    A = tf.math.softmax(A)  # softmax on attention scores

    if args.att_only:
        return att_score

    if args.mil_ins:
        ins_labels, ins_logits_unnorm, ins_logits = ins_call(
            m_ins_classifier=ins_net,
            bag_label=slide_label,
            h=h,
            A=A,
            args=args,
        )

    slide_score_unnorm, Y_hat, Y_prob, predict_slide_label, Y_true = s_bag_call(
        bag_classifier=bag_net, bag_label=slide_label, A=A, h=h, n_class=args.n_class
    )

    return (
        att_score,
        A,
        h,
        ins_labels,
        ins_logits_unnorm,
        ins_logits,
        slide_score_unnorm,
        Y_prob,
        Y_hat,
        Y_true,
        predict_slide_label,
    )


def m_clam_call(
    att_net,
    ins_net,
    bag_net,
    img_features,
    slide_label,
    args,
):
    """_summary_

    Args:
        att_net (_type_): _description_
        ins_net (_type_): _description_
        bag_net (_type_): _description_
        img_features (_type_): _description_
        slide_label (_type_): _description_
        args (_type_): _description_

    Returns:
        _type_: _description_
    """
    if args.att_gate:
        h, A = g_att_call(g_att_net=att_net, img_features=img_features)
    else:
        h, A = ng_att_call(ng_att_net=att_net, img_features=img_features)
    att_score = A  # output from attention network
    A = tf.math.softmax(A)  # softmax on attention scores

    if args.att_only:
        return att_score

    if args.mil_ins:
        ins_labels, ins_logits_unnorm, ins_logits = ins_call(
            m_ins_classifier=ins_net,
            bag_label=slide_label,
            h=h,
            A=A,
            args=args,
        )

    slide_score_unnorm, Y_hat, Y_prob, predict_slide_label, Y_true = m_bag_call(
        m_bag_classifier=bag_net,
        bag_label=slide_label,
        A=A,
        h=h,
        args=args,
    )

    return (
        att_score,
        A,
        h,
        ins_labels,
        ins_logits_unnorm,
        ins_logits,
        slide_score_unnorm,
        Y_prob,
        Y_hat,
        Y_true,
        predict_slide_label,
    )


def model_save(
    c_model,
    args,
):
    """_summary_

    Args:
        c_model (_type_): _description_
        args (_type_): _description_
    """
    model_checkpoint_path = os.path.join(args.checkpoints_dir, "models")
    os.makedirs(model_checkpoint_path, exist_ok=True)

    clam_model_names = ["_Att", "_Ins", "_Bag"]

    if args.m_clam_op:
        if args.att_gate:
            att_nets = c_model.clam_model()[0]
            for m in range(len(att_nets)):
                att_nets[m].save(
                    os.path.join(
                        model_checkpoint_path,
                        "G" + clam_model_names[0],
                        "Model_" + str(m + 1),
                    )
                )
        else:
            att_nets = c_model.clam_model()[0]
            for m in range(len(att_nets)):
                att_nets[m].save(
                    os.path.join(
                        model_checkpoint_path,
                        "NG" + clam_model_names[0],
                        "Model_" + str(m + 1),
                    )
                )

        for n in range(args.n_class):
            ins_nets = c_model.clam_model()[1]
            bag_nets = c_model.clam_model()[2]

            ins_nets[n].save(
                os.path.join(
                    model_checkpoint_path, "M" + clam_model_names[1], "Class_" + str(n)
                )
            )
            bag_nets[n].save(
                os.path.join(
                    model_checkpoint_path, "M" + clam_model_names[2], "Class_" + str(n)
                )
            )
    else:
        if args.att_gate:
            att_nets = c_model.clam_model()[0]
            for m in range(len(att_nets)):
                att_nets[m].save(
                    os.path.join(
                        model_checkpoint_path,
                        "G" + clam_model_names[0],
                        "Model_" + str(m + 1),
                    )
                )
        else:
            att_nets = c_model.clam_model()[0]
            for m in range(len(att_nets)):
                att_nets[m].save(
                    os.path.join(
                        model_checkpoint_path,
                        "NG" + clam_model_names[0],
                        "Model_" + str(m + 1),
                    )
                )

        for n in range(args.n_class):
            ins_nets = c_model.clam_model()[1]
            ins_nets[n].save(
                os.path.join(
                    model_checkpoint_path, "M" + clam_model_names[1], "Class_" + str(n)
                )
            )

        c_model.clam_model()[2].save(
            os.path.join(model_checkpoint_path, "S" + clam_model_names[2])
        )


def restore_model(
    args,
):
    """_summary_

    Args:
        args (_type_): _description_

    Returns:
        _type_: _description_
    """
    model_checkpoint_path = os.path.join(args.checkpoints_dir, "models")
    assert os.path.exists(model_checkpoint_path), Exception(
        f"Not Found Error: Could not find the model checkpoint path on {model_checkpoint_path}"
    )

    clam_model_names = ["_Att", "_Ins", "_Bag"]

    trained_att_net = list()
    trained_ins_classifier = list()
    trained_bag_classifier = list()

    if args.m_clam_op:
        if args.att_gate:
            att_nets_dir = os.path.join(
                model_checkpoint_path, "G" + clam_model_names[0]
            )
            for k in range(len(os.listdir(att_nets_dir))):
                att_net = tf.keras.models.load_model(
                    os.path.join(att_nets_dir, "Model_" + str(k + 1))
                )
                trained_att_net.append(att_net)
        else:
            att_nets_dir = os.path.join(
                model_checkpoint_path, "NG" + clam_model_names[0]
            )
            for k in range(len(os.listdir(att_nets_dir))):
                att_net = tf.keras.models.load_model(
                    os.path.join(att_nets_dir, "Model_" + str(k + 1))
                )
                trained_att_net.append(att_net)

        ins_nets_dir = os.path.join(model_checkpoint_path, "M" + clam_model_names[1])
        bag_nets_dir = os.path.join(model_checkpoint_path, "M" + clam_model_names[2])

        for m in range(args.n_class):
            ins_net = tf.keras.models.load_model(
                os.path.join(ins_nets_dir, "Class_" + str(m))
            )
            bag_net = tf.keras.models.load_model(
                os.path.join(bag_nets_dir, "Class_" + str(m))
            )

            trained_ins_classifier.append(ins_net)
            trained_bag_classifier.append(bag_net)

        c_trained_model = [
            trained_att_net,
            trained_ins_classifier,
            trained_bag_classifier,
        ]
    else:
        if args.att_gate:
            att_nets_dir = os.path.join(
                model_checkpoint_path, "G" + clam_model_names[0]
            )
            for k in range(len(os.listdir(att_nets_dir))):
                att_net = tf.keras.models.load_model(
                    os.path.join(att_nets_dir, "Model_" + str(k + 1))
                )
                trained_att_net.append(att_net)
        else:
            att_nets_dir = os.path.join(
                model_checkpoint_path, "NG" + clam_model_names[0]
            )
            for k in range(len(os.listdir(att_nets_dir))):
                att_net = tf.keras.models.load_model(
                    os.path.join(att_nets_dir, "Model_" + str(k + 1))
                )
                trained_att_net.append(att_net)

        ins_nets_dir = os.path.join(model_checkpoint_path, "M" + clam_model_names[1])

        for m in range(args.n_class):
            ins_net = tf.keras.models.load_model(
                os.path.join(ins_nets_dir, "Class_" + str(m))
            )
            trained_ins_classifier.append(ins_net)

        bag_nets_dir = os.path.join(model_checkpoint_path, "S" + clam_model_names[2])
        trained_bag_classifier.append(tf.keras.models.load_model(bag_nets_dir))

        c_trained_model = [
            trained_att_net,
            trained_ins_classifier,
            trained_bag_classifier[0],
        ]

    return c_trained_model
