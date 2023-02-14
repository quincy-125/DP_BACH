from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
import os
import sys
import io
import tensorflow as tf

# import matplotlib.pylab as plt
import numpy as np
from PIL import Image
import glob

# output_dir="/projects/shart/digital_pathology/data/TCGA_MAYO/FINAL_TF/tfrecord_brca2_level3_img"
# input_dir='/projects/shart/digital_pathology/data/TCGA_MAYO/FINAL_TF/tfrecord_brca2_level3'
input_dir = sys.argv[1]
output_dir = sys.argv[2]
input_dir = input_dir.strip()
output_dir = output_dir.strip()
# print(output_dir)
# print(input_dir)
# sys.exit(0)
os.mkdir(output_dir + "/train")
os.mkdir(output_dir + "/train/1")
os.mkdir(output_dir + "/train/0")
os.mkdir(output_dir + "/val")
os.mkdir(output_dir + "/val/1")
os.mkdir(output_dir + "/val/0")
os.mkdir(output_dir + "/test")
os.mkdir(output_dir + "/test/1")
os.mkdir(output_dir + "/test/0")
train_files = glob.glob(input_dir + "/train/*/*tfrecords")
val_files = glob.glob(input_dir + "/val/*/*tfrecords")
test_files = glob.glob(input_dir + "/test/*/*tfrecords")
files = train_files + val_files + test_files

num = 0
for i in files:
    cd = os.path.dirname(i)
    mut = os.path.basename(cd)
    cd = os.path.dirname(cd)
    strval = os.path.basename(cd)
    # print(i,mut,strval)
    # sys.exit(0)
    raw_dataset = tf.data.TFRecordDataset(i)
    for raw_record in raw_dataset:
        example = tf.train.Example()
        # print(raw_record)
        example.ParseFromString(raw_record.numpy())
        # result = tf.train.Example.ParseFromString(example.numpy())
        num += 1
        for k, v in example.features.feature.items():
            # print(k)
            if k == "image/encoded":
                # print(k, "Skipping...")
                stream = io.BytesIO(v.bytes_list.value[0])
                file_out = Image.open(stream).convert("RGB")
                # fileout = im.convert("RGB", fileout)
            if k == "image/name":
                filename = v.bytes_list.value[0]
                fn = filename.decode("utf-8")
                fn = fn.replace(".png", ".jpg")
        file_out.save(output_dir + "/" + strval + "/" + mut + "/" + fn, "jpeg")
        # print(output_dir+'/'+strval+'/'+mut+'/'+fn)
        # sys.exit(0)
