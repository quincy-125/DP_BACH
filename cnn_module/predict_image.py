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
"""Loadmodel"""
new_model = models.load_model(model_path)

"""reading the file paths"""
fobj = open(input_file)

IMAGE_SHAPE = (256, 256)

for i in fobj:
    ori_label = os.path.basename(os.path.dirname(i))
    ori_cate = os.path.basename(os.path.dirname(os.path.dirname(i)))
    i = i.strip()
    file_out = Image.open(i)
    file_out = np.asarray(file_out)
    file_out = (file_out / 127.5) - 1
    file_out = np.reshape(file_out, (1, 256, 256, 3))
    result = np.asarray(new_model.predict(file_out))
    pred_label = 0
    if float(result[0][1]) > 0.5:
        pred_label = 1

    # print(result)
    print(ori_cate + " " + model_name + " " + ori_label + " " + str(pred_label))
