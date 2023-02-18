from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
import os
import sys
import io
import tensorflow as tf
from callbacks import CallBacks
from model_factory import GetModel
from preprocess import Preprocess, format_example
import matplotlib.pylab as plt
from tensorflow.keras import layers
from tensorflow.keras import models
import numpy as np
from PIL import Image
import glob

# filepath="/projects/shart/digital_pathology/results/tcga_pten_General-ImageClassifier/train/ResNet50_SGD_0.001-BinaryCrossentropy/my_model.h5"
# filepath='/projects/shart/digital_pathology/results/General-ImageClassifier/tcga_pten_General-ImageClassifier/train/ResNet50_SGD_0.001-BinaryCrossentropy/my_model.h5'
# input_dir='/projects/shart/digital_pathology/data/TCGA/General-ImageClassifier-Level3_selected_PTEN/train'
# input_dir='/projects/shart/digital_pathology/data/TCGA/General-ImageClassifier-Level3_selected_PTEN/val'
# files=glob.glob(input_dir+'/*/*png')

# new_model = models.load_model(filepath)
# IMAGE_SHAPE = (256, 256)
# #new_model.summary()
# for file in files:
# #print(file)
# #sys.exit(0)
# file_out = Image.open(file)
# file_out = file_out.resize(IMAGE_SHAPE)
# file_out=np.asarray(file_out)
# file_out = np.reshape(file_out,(1,256,256,3))
# result = np.asarray(new_model.predict(file_out))
# #print(result)
# print(os.path.basename(file)+' '+str((result[0][0]))+' '+str((result[0][1])))
# #sys.exit(0)

filepath = "/projects/shart/digital_pathology/results/General-ImageClassifier/tcga_brca1_General-ImageClassifier/train/ResNet50_SGD_0.001-BinaryCrossentropy/my_model.h5"
input_dir = "/projects/shart/digital_pathology/data/TCGA/General-ImageClassifier-Level3_selected_BRCA1"
output_file = "predict_brca1.txt"
myfile = open(output_file, mode="wt")
new_model = models.load_model(filepath)
IMAGE_SHAPE = (256, 256)
train_files = glob.glob(input_dir + "/train/*/*tfrecords")
val_files = glob.glob(input_dir + "/val/*/*tfrecords")
files = train_files + val_files
for i in files:
    cd = os.path.dirname(i)
    mut = os.path.basename(cd)
    cd = os.path.dirname(cd)
    strval = os.path.basename(cd)

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
                # print(file_out.shape)
                # sys.exit(0)
                file_out = np.reshape(file_out, (1, 256, 256, 3))
                result = np.asarray(new_model.predict(file_out))
            if k == "image/name":
                filename = v.bytes_list.value[0]
        finalstr = filename.decode("utf-8") + "." + mut + "." + strval + ".png"
        myfile.write(
            finalstr + " " + str((result[0][0])) + " " + str((result[0][1])) + "\n"
        )
myfile.close()
