from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
import os
import sys
import tensorflow as tf
import matplotlib.pylab as plt
from tensorflow.keras import layers
from tensorflow.keras import models
import numpy as np
import PIL.Image as Image
import glob
from preprocess import Preprocess, format_example, format_example_tf, update_status

input_file_path = sys.argv[1]
input_file_path = input_file_path.strip()
model_path = sys.argv[2]
model_path = model_path.strip()
ori_cate = sys.argv[3]
ori_cate = ori_cate.strip()
model_name = os.path.basename(os.path.dirname(model_path))
"""Loadmodel"""
new_model = models.load_model(model_path)

"""reading the file paths"""
IMAGE_SHAPE = (256, 256)

classes = sorted(os.listdir(input_file_path))
files_list = []
labels_list = []
for x in classes:
    class_files = os.listdir(os.path.join(input_file_path, x))
    class_files = [os.path.join(input_file_path, x, j) for j in class_files]
    class_labels = [int(x) for y in class_files]
    files_list = files_list + class_files
    labels_list = labels_list + class_labels
update_status(False)
t_path_ds = tf.data.Dataset.from_tensor_slices(files_list)
t_image_ds = t_path_ds.map(
    format_example, num_parallel_calls=tf.data.experimental.AUTOTUNE
)
t_label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels_list, tf.int64))
t_image_label_ds = tf.data.Dataset.zip((t_image_ds, t_label_ds, t_path_ds))
train_ds = t_image_label_ds.batch(64).prefetch(
    buffer_size=tf.data.experimental.AUTOTUNE
)
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    for step, (image, label, file) in enumerate(train_ds):
        # print(step,image.shape,label.shape,file.shape)
        lst_label = list(label.numpy())
        lst_file = list(file.numpy())
        result = np.asarray(new_model.predict_on_batch(image))
        lst_result = list(result)
        for i in range(0, len(lst_label)):
            print(
                model_name,
                os.path.basename(lst_file[i].decode("utf-8")),
                lst_label[i],
                ori_cate,
                lst_result[i][0],
                lst_result[i][1],
            )
    # sys.exit(0)
