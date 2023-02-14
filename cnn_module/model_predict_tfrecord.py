import sys
import random
import os
import argparse
import sys
import pwd
import time
import subprocess
import re
import shutil
from sklearn import metrics
import re
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
import glob

import logging
import io
import tensorflow as tf

# from callbacks import CallBacks
# from model_factory import GetModel
# from preprocess import Preprocess, format_example
import matplotlib.pylab as plt
from tensorflow.keras import layers
from tensorflow.keras import models
import numpy as np
from PIL import Image
import os

dir = sys.argv[1]
dir = dir.strip()
model_path = sys.argv[2]
model_path = model_path.strip()
model_name = sys.argv[3]
model_name = model_name.strip()

"""Loadmodel"""
new_model = models.load_model(model_path)

test_dir = os.path.join(dir, "test")
train_dir = os.path.join(dir, "train")
val_dir = os.path.join(dir, "val")
test_dir_files = glob.glob(test_dir + "/*/*.tfrecords")
train_dir_files = glob.glob(train_dir + "/*/*.tfrecords")
val_dir_files = glob.glob(val_dir + "/*/*.tfrecords")
# print(dir,test_dir,train_dir,val_dir,len(test_dir_files),len(train_dir_files),len(val_dir_files))
IMAGE_SHAPE = (256, 256)

for i in test_dir_files:
    # print(i)
    ori_label = os.path.basename(os.path.dirname(i))
    raw_dataset = tf.data.TFRecordDataset(i)
    for raw_record in raw_dataset:
        example = tf.train.Example()
        # print(raw_record)
        example.ParseFromString(raw_record.numpy())
        # result = tf.train.Example.ParseFromString(example.numpy())
        for k, v in example.features.feature.items():
            # print(k)
            if k == "image/encoded":
                # print(k, "Skipping...")
                stream = io.BytesIO(v.bytes_list.value[0])
                file_out = Image.open(stream)
                # file_out.save("sampletf.png", "png")

                fileout = file_out.resize(IMAGE_SHAPE).convert("RGB")
                # fileout = im.convert("RGB", fileout)
                file_out = np.asarray(fileout)
                file_out = (file_out / 127.5) - 1
                file_out = np.reshape(file_out, (1, 256, 256, 3))
                result = np.asarray(new_model.predict(file_out))
            if k == "image/name":
                filename = v.bytes_list.value[0]
        finalstr = filename.decode("utf-8") + " " + ori_label + " test"
        print(
            model_name
            + " "
            + finalstr
            + " "
            + str((result[0][0]))
            + " "
            + str((result[0][1]))
            + "\n"
        )


for i in train_dir_files:
    # print(i)
    ori_label = os.path.basename(os.path.dirname(i))
    raw_dataset = tf.data.TFRecordDataset(i)
    for raw_record in raw_dataset:
        example = tf.train.Example()
        # print(raw_record)
        example.ParseFromString(raw_record.numpy())
        # result = tf.train.Example.ParseFromString(example.numpy())
        for k, v in example.features.feature.items():
            # print(k)
            if k == "image/encoded":
                # print(k, "Skipping...")
                stream = io.BytesIO(v.bytes_list.value[0])
                file_out = Image.open(stream)
                # file_out.save("sampletf.png", "png")

                fileout = file_out.resize(IMAGE_SHAPE).convert("RGB")
                # fileout = im.convert("RGB", fileout)
                file_out = np.asarray(fileout)
                file_out = (file_out / 127.5) - 1
                file_out = np.reshape(file_out, (1, 256, 256, 3))
                result = np.asarray(new_model.predict(file_out))
            if k == "image/name":
                filename = v.bytes_list.value[0]
        finalstr = filename.decode("utf-8") + " " + ori_label + " train"
        print(
            model_name
            + " "
            + finalstr
            + " "
            + str((result[0][0]))
            + " "
            + str((result[0][1]))
            + "\n"
        )

for i in val_dir_files:
    # print(i)
    ori_label = os.path.basename(os.path.dirname(i))
    raw_dataset = tf.data.TFRecordDataset(i)
    for raw_record in raw_dataset:
        example = tf.train.Example()
        # print(raw_record)
        example.ParseFromString(raw_record.numpy())
        # result = tf.train.Example.ParseFromString(example.numpy())
        for k, v in example.features.feature.items():
            # print(k)
            if k == "image/encoded":
                # print(k, "Skipping...")
                stream = io.BytesIO(v.bytes_list.value[0])
                file_out = Image.open(stream)
                # file_out.save("sampletf.png", "png")

                fileout = file_out.resize(IMAGE_SHAPE).convert("RGB")
                # fileout = im.convert("RGB", fileout)
                file_out = np.asarray(fileout)
                file_out = (file_out / 127.5) - 1
                file_out = np.reshape(file_out, (1, 256, 256, 3))
                result = np.asarray(new_model.predict(file_out))
            if k == "image/name":
                filename = v.bytes_list.value[0]
        finalstr = filename.decode("utf-8") + " " + ori_label + " val"
        print(
            model_name
            + " "
            + finalstr
            + " "
            + str((result[0][0]))
            + " "
            + str((result[0][1]))
        )
