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
from preprocess import Preprocess, format_example, update_status
import random
import datetime

input_file_path = sys.argv[1]
input_file_path = input_file_path.strip()
model_path = sys.argv[2]
model_path = model_path.strip()
batch_size = sys.argv[3]
batch_size = int(batch_size.strip())
output_file = sys.argv[4]
output_file = output_file.strip()
"""Loadmodel"""
new_model = models.load_model(model_path)

"""reading the file paths"""
IMAGE_SHAPE = (256, 256)

train_path = os.path.join(input_file_path, "train")
val_path = os.path.join(input_file_path, "val")
test_path = os.path.join(input_file_path, "test")

classes = sorted(os.listdir(train_path))
"""getting training & label files"""
files_list = []
labels_list = []
for x in classes:
    class_files = os.listdir(os.path.join(train_path, x))
    class_files = [os.path.join(train_path, x, j) for j in class_files]
    class_labels = [int(x) for y in class_files]
    files_list = files_list + class_files
    labels_list = labels_list + class_labels
train_files_list = files_list
train_labels_list = labels_list


"""getting validation & label files"""
files_list = []
labels_list = []
for x in classes:
    class_files = os.listdir(os.path.join(val_path, x))
    class_files = [os.path.join(val_path, x, j) for j in class_files]
    class_labels = [int(x) for y in class_files]
    files_list = files_list + class_files
    labels_list = labels_list + class_labels
val_files_list = files_list
val_labels_list = labels_list


"""getting test & label files"""
files_list = []
labels_list = []
for x in classes:
    class_files = os.listdir(os.path.join(test_path, x))
    class_files = [os.path.join(test_path, x, j) for j in class_files]
    class_labels = [int(x) for y in class_files]
    files_list = files_list + class_files
    labels_list = labels_list + class_labels
test_files_list = files_list
test_labels_list = labels_list

"""Reference sets"""
files_list = []
labels_list = []
for x in classes:
    index = [i for i in range(len(train_labels_list)) if train_labels_list[i] == int(x)]
    ind = [random.randint(0, len(index)) for i in range(8)]
    label = [train_labels_list[index[i]] for i in ind]
    labels_list = labels_list + label
    file = [train_files_list[index[i]] for i in ind]
    files_list = files_list + file

"""training data pairs"""
list_img_index1 = []
list_img_index2 = []
list_label_index = []
list_lbls = []
category = []
# for i in range(len(train_files_list)):
#    for j in range(len(files_list)):
#        list_img_index1.append(train_files_list[i])
#        list_img_index2.append(files_list[j])
#        list_label_index.append([train_labels_list[i],labels_list[j]])
#        if train_labels_list[i] == labels_list[j]:
#            list_lbls.append(0)
#        else:
#            list_lbls.append(1)
#        category.append("train")

"""validation data pairs"""
for i in range(len(val_files_list)):
    for j in range(len(files_list)):
        list_img_index1.append(val_files_list[i])
        list_img_index2.append(files_list[j])
        list_label_index.append([val_labels_list[i], labels_list[j]])
        if val_labels_list[i] == labels_list[j]:
            list_lbls.append(0)
        else:
            list_lbls.append(1)
        category.append("val")

"""testing data pairs"""
for i in range(len(test_files_list)):
    for j in range(len(files_list)):
        list_img_index1.append(test_files_list[i])
        list_img_index2.append(files_list[j])
        list_label_index.append([test_labels_list[i], labels_list[j]])
        if test_labels_list[i] == labels_list[j]:
            list_lbls.append(0)
        else:
            list_lbls.append(1)
        category.append("test")

labels = tf.one_hot(list_lbls, 2)
update_status(False)
num_im = int(len(category) / 64)
t_category = tf.data.Dataset.from_tensor_slices(category)
t_path_ds1 = tf.data.Dataset.from_tensor_slices(list_img_index1)
t_image_ds1 = t_path_ds1.map(
    format_example, num_parallel_calls=tf.data.experimental.AUTOTUNE
)
t_path_ds2 = tf.data.Dataset.from_tensor_slices(list_img_index2)
t_image_ds2 = t_path_ds2.map(
    format_example, num_parallel_calls=tf.data.experimental.AUTOTUNE
)
t_label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.int64))
t_label_ds_ori = tf.data.Dataset.from_tensor_slices(tf.cast(list_label_index, tf.int64))

t_image_label_ds = tf.data.Dataset.zip(
    (
        t_image_ds1,
        t_image_ds2,
        t_label_ds,
        t_label_ds_ori,
        t_path_ds1,
        t_path_ds2,
        t_category,
    )
)
train_ds = t_image_label_ds.batch(64).prefetch(
    buffer_size=tf.data.experimental.AUTOTUNE
)
final_arr = []
final_dict = {}
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    for step, (image1, image2, label, label_ori, file1, file2, cate) in enumerate(
        train_ds
    ):
        # if step>5:
        #    break
        if step % 100 == 0:
            print(step, "/", num_im, datetime.datetime.now())
        lst_cate = list(cate.numpy())
        lst_label = list(label.numpy())
        lst_label_ori = list(label_ori.numpy())
        lst_file1 = list(file1.numpy())
        lst_file2 = list(file2.numpy())
        result = np.asarray(new_model.predict_on_batch([image1, image2]))
        lst_result = list(result)
        for i in range(0, len(lst_label)):
            # print(step, lst_file1[i],lst_file2[i],lst_label[i],lst_label_ori[i],lst_result[i][0],lst_result[i][1])
            # print(lst_file1[i].decode("utf-8"),lst_label_ori[i][0],lst_label_ori[i][1],lst_result[i][0])
            # final_arr.append([lst_file1[i].decode("utf-8"),lst_label_ori[i][0],lst_label_ori[i][1],lst_result[i][0]])
            elem = (
                lst_file1[i].decode("utf-8")
                + " "
                + str(lst_label_ori[i][0])
                + " "
                + lst_cate[i].decode("utf-8")
            )
            if not elem in final_arr:
                final_arr.append(elem)
            key = elem + " " + str(lst_label_ori[i][1])
            if not key in final_dict:
                final_dict[key] = [lst_result[i][0]]
            else:
                final_dict[key].append(lst_result[i][0])
myfile = open(output_file, mode="wt")
myfile.write("Filename ori_label category softmax_0 softmax_1\n")
for i in final_arr:
    class_0 = np.mean(final_dict[i + " 0"])
    class_1 = np.mean(final_dict[i + " 1"])
    softmax_0 = class_0 / (class_0 + class_1)
    softmax_1 = class_1 / (class_0 + class_1)
    myfile.write(i + " " + str(softmax_0) + " " + str(softmax_1) + "\n")
    # print(i,class_0,class_1)
myfile.close()
