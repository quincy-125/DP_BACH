from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
import os
import sys
import tensorflow as tf

from preprocess import Preprocess
from data_runner import DataRunner
from model_factory import GetModel
from callbacks import CallBacks

from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Flatten
from tensorflow.keras import Model
from losses import triplet_loss

import matplotlib.pylab as plt
from tensorflow.keras import layers
from tensorflow.keras import models
import numpy as np
import PIL.Image as Image
import glob

filepath = "/projects/shart/digital_pathology/data/biliary_2020/annotations/images/Images_QC/sample_images/triplet_lossless_tfrecord_img/ResNet50_RMSprop_0.001/my_model.h5"
input_dir = "/projects/shart/digital_pathology/data/biliary_2020/annotations/images/Images_QC/sample_images/test/0"
files0 = glob.glob(input_dir + "/*jpg")
input_dir = "/projects/shart/digital_pathology/data/biliary_2020/annotations/images/Images_QC/sample_images/test/1"
files1 = glob.glob(input_dir + "/*jpg")
# print(files)
# sys.exit(0)
new_model = models.load_model(filepath, custom_objects={"triplet_loss": triplet_loss})
IMAGE_SHAPE = (256, 256)
img_size = 256
# new_model.summary()
i = 0
for file in files0:
    image = tf.io.read_file(file)
    # image = tf.io.decode_png(image)
    image = tf.io.decode_jpeg(image)
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)
    image = tf.image.resize(image, (img_size, img_size))
    file_out = tf.reshape(image, (1, img_size, img_size, 3))
    if i == len(files1):
        i = 0
    file1 = files1[i]
    i = i + 1
    image = tf.io.read_file(file1)
    # image = tf.io.decode_png(image)
    image = tf.io.decode_jpeg(image)
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)
    image = tf.image.resize(image, (img_size, img_size))
    file_out1 = tf.reshape(image, (1, img_size, img_size, 3))

    result = np.asarray(new_model.predict([file_out, file_out, file_out]))
    print(result)
    print(file + " " + file1 + " " + str((result[0][0])) + " " + str((result[0][1])))

    result = np.asarray(new_model.predict([file_out, file_out, file_out1]))
    print(result)
    print(file + " " + file1 + " " + str((result[0][0])) + " " + str((result[0][1])))

    result = np.asarray(new_model.predict([file_out1, file_out1, file_out]))
    print(result)
    print(file + " " + file1 + " " + str((result[0][0])) + " " + str((result[0][1])))

    result = np.asarray(new_model.predict([file_out1, file_out1, file_out1]))
    print(result)
    print(file + " " + file1 + " " + str((result[0][0])) + " " + str((result[0][1])))
    # sys.exit(0)
