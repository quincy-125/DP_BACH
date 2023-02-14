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


input_file = sys.argv[1]
input_file = input_file.strip()
model_path = sys.argv[2]
model_path = model_path.strip()
# model_name=sys.argv[3]
# model_name=model_name.strip()
model_name = os.path.basename(os.path.dirname(model_path))
model_name1 = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
model_name = model_name1 + "/" + model_name
"""Loadmodel"""
new_model = models.load_model(model_path)

"""reading the file paths"""
fobj = open(input_file)

IMAGE_SHAPE = (256, 256)
img_size = 256
# DenseNet201_RMSprop_1e-05-BinaryCrossentropy 12.WT.clean.tiff_x_16384_24576_y_57344_65536_150.86_46.62_203366.png 0 test 0.27820253 0.72179747
for i in fobj:
    patch_name = os.path.basename(i).strip()
    ori_label = os.path.basename(os.path.dirname(i))
    ori_cate = os.path.basename(os.path.dirname(os.path.dirname(i)))
    i = i.strip()

    image = tf.io.read_file(i)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32) / 255
    image = tf.image.resize(image, (img_size, img_size))
    # image = tf.image.per_image_standardization(image)
    image = tf.reshape(image, (1, img_size, img_size, 3))

    # file_out = Image.open(i)
    # file_out=np.asarray(file_out)
    # file_out = tf.convert_to_tensor(file_out)
    # file_out = tf.image.per_image_standardization(file_out)
    # file_out = file_out.numpy()
    # #file_out = (file_out/255)
    # file_out = np.reshape(file_out,(1,256,256,3))
    result = np.asarray(new_model.predict(image))
    pred_label = 0
    if float(result[0][1]) > 0.5:
        pred_label = 1

    print(
        model_name
        + " "
        + patch_name
        + " "
        + ori_label
        + " "
        + ori_cate
        + " "
        + str(result[0][0])
        + " "
        + str(result[0][1])
    )
    # sys.exit(0)
    # print(ori_cate+' '+model_name+' '+ori_label+' '+str(pred_label))
