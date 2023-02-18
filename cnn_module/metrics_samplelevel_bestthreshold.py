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

# import pandas as pd

input_softmax_file = sys.argv[1]
input_softmax__file = input_softmax_file.strip()
model_name = sys.argv[2]
model_name = model_name.strip()


def report(fish_list, result_list, name):
    tn, fp, fn, tp = metrics.confusion_matrix(fish_list, result_list).ravel()
    tn = int(tn)
    fp = int(fp)
    fn = int(fn)
    tp = int(tp)
    # sys.exit(0)
    sensitivity = round(tp / (tp + fn), 2)
    # print(sensitivity)
    # sys.exit(0)
    specificity = round(tn / (tn + fp), 2)
    accuracy = round((tp + tn) / (tn + fp + fn + tp), 2)
    fpr, tpr, thresholds = metrics.roc_curve(fish_list, result_list, pos_label=1)
    auc = round(metrics.auc(fpr, tpr), 2)
    # print(name , "TN:",tn, "FP:",fp, "FN:",fn, "TP:",tp, "Sensitivity:",sensitivity, "Specificity:",specificity, "AUC:",auc,"Accuracy:",accuracy)
    print(name, tn, fp, fn, tp, sensitivity, specificity, auc, accuracy)


patch_name_col = 2
category = 4
label_col = 3
softmax_1_col = 6
# array values
patch_name_col = patch_name_col - 1
category = category - 1
softmax_1_col = softmax_1_col - 1
label_col = label_col - 1
# reading input softmax file
fobj = open(input_softmax_file)
# myfile = open(output, mode='wt')
fobj.readline()
sample = []
cate = []
softmax_1 = []
label = []
for file in fobj:
    file = file.strip()
    p = file.split(" ")
    samp_arr = p[patch_name_col].split(".")
    # samp_nm = samp_arr[0]+'.'+samp_arr[1]+'.'+samp_arr[2]
    samp_nm = samp_arr[0]
    sample.append(samp_nm)
    cate.append(p[category])
    label.append(int(p[label_col]))
    softmax_1.append(float(p[softmax_1_col]))

# #subset df by train
train_index = [i for i in range(len(cate)) if cate[i] == "train"]
train_label = [label[train_index[i]] for i in range(len(train_index))]
train_softmax = [softmax_1[train_index[i]] for i in range(len(train_index))]
softmax_range = [i for i in np.arange(0.01, 1, 0.01)]
prob_aucs = [
    roc_auc_score(
        train_label, [1 if x >= softmax_range[i] else 0 for x in train_softmax]
    )
    for i in range(len(softmax_range))
]
# print(prob_aucs)
# getting the index for best auc
idx_opti = np.argmax(prob_aucs)
# best threshold
threshold = softmax_range[idx_opti]
print("Best threshold", threshold)
print("name", "tn", "fp", "fn", "tp", "sensitivity", "specificity", "auc", "accuracy")
# Get metrics for train
train_samples = [sample[train_index[i]] for i in range(len(train_index))]
train_samples_uniq = list(set(train_samples))
indexes = [train_samples.index(x) for x in set(train_samples)]
train_samples_uniq_label = [train_label[indexes[i]] for i in range(len(indexes))]
train_samples_uniq_label_softmax = [
    np.mean(
        [
            train_softmax[x]
            for x in range(len(train_samples))
            if train_samples[x] == train_samples_uniq[i]
        ]
    )
    for i in range(len(train_samples_uniq))
]
train_samples_uniq_label_pred = [
    1 if i > threshold else 0 for i in train_samples_uniq_label_softmax
]
report(train_samples_uniq_label, train_samples_uniq_label_pred, "train")
# Get metrics for val
val_index = [i for i in range(len(cate)) if cate[i] == "val"]
val_label = [label[val_index[i]] for i in range(len(val_index))]
val_softmax = [softmax_1[val_index[i]] for i in range(len(val_index))]
val_samples = [sample[val_index[i]] for i in range(len(val_index))]
val_samples_uniq = list(set(val_samples))
indexes = [val_samples.index(x) for x in set(val_samples)]
val_samples_uniq_label = [val_label[indexes[i]] for i in range(len(indexes))]
val_samples_uniq_label_softmax = [
    np.median(
        [
            val_softmax[x]
            for x in range(len(val_samples))
            if val_samples[x] == val_samples_uniq[i]
        ]
    )
    for i in range(len(val_samples_uniq))
]
val_samples_uniq_label_pred = [
    1 if i > threshold else 0 for i in val_samples_uniq_label_softmax
]
report(val_samples_uniq_label, val_samples_uniq_label_pred, "val")
# Get metrics for test
test_index = [i for i in range(len(cate)) if cate[i] == "test"]
test_label = [label[test_index[i]] for i in range(len(test_index))]
test_softmax = [softmax_1[test_index[i]] for i in range(len(test_index))]
test_samples = [sample[test_index[i]] for i in range(len(test_index))]
test_samples_uniq = list(set(test_samples))
indexes = [test_samples.index(x) for x in set(test_samples)]
test_samples_uniq_label = [test_label[indexes[i]] for i in range(len(indexes))]
test_samples_uniq_label_softmax = [
    np.mean(
        [
            test_softmax[x]
            for x in range(len(test_samples))
            if test_samples[x] == test_samples_uniq[i]
        ]
    )
    for i in range(len(test_samples_uniq))
]
test_samples_uniq_label_pred = [
    1 if i > threshold else 0 for i in test_samples_uniq_label_softmax
]


report(test_samples_uniq_label, test_samples_uniq_label_pred, "test")
