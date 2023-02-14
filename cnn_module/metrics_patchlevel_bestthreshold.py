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


print("name", "tn", "fp", "fn", "tp", "sensitivity", "specificity", "auc", "accuracy")
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
cate = []
softmax_1 = []
label = []
for file in fobj:
    file = file.strip()
    p = file.split(" ")
    cate.append(p[category])
    label.append(int(p[label_col]))
    softmax_1.append(float(p[softmax_1_col]))

# #subset df by train
train_index = [i for i in range(len(cate)) if cate[i] == "train"]
train_label = [label[train_index[i]] for i in range(len(train_index))]
train_softmax = [softmax_1[train_index[i]] for i in range(len(train_index))]
softmax_range = [i for i in np.arange(0.01, 1, 0.01)]
print("Calculating prob aucs")
prob_aucs = [
    roc_auc_score(
        train_label, [1 if x >= softmax_range[i] else 0 for x in train_softmax]
    )
    for i in range(len(softmax_range))
]
print(prob_aucs)
# getting the index for best auc
idx_opti = np.argmax(prob_aucs)
# best threshold
threshold = softmax_range[idx_opti]
print(threshold)
# Get metrics for train
train_ori = []
train_pred = []
test_ori = []
test_pred = []
val_ori = []
val_pred = []
for i in range(len(softmax_1)):
    val = 0
    if softmax_1[i] > threshold:
        val = 1
    if cate[i] == "train":
        train_ori.append(label[i])
        train_pred.append(val)
    if cate[i] == "test":
        test_ori.append(label[i])
        test_pred.append(val)
    if cate[i] == "val":
        val_ori.append(label[i])
        val_pred.append(val)
fobj.close()
report(test_ori, test_pred, model_name + " test")
report(train_ori, train_pred, model_name + " train")
report(val_ori, val_pred, model_name + " val")
