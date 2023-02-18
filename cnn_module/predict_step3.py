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


# print(os.environ.get('CUDA_VISIBLE_DEVICES'))
# sys.exit(0)
def report(fish_list, result_list, name):
    tn, fp, fn, tp = metrics.confusion_matrix(fish_list, result_list).ravel()
    tn = int(tn)
    fp = int(fp)
    fn = int(fn)
    tp = int(tp)
    # print(name,tn, fp, fn, tp)
    # tn = len([ i for i in range(0,len(fish_list),1) if fish_list[i]==0  and result_list[i]==0] )
    # tp = len([ i for i in range(0,len(fish_list),1) if fish_list[i]==1  and result_list[i]==1] )
    # fn = len([ i for i in range(0,len(fish_list),1) if fish_list[i]==1  and result_list[i]==0] )
    # fp = len([ i for i in range(0,len(fish_list),1) if fish_list[i]==0  and result_list[i]==1] )
    # print(name,tn, fp, fn, tp)
    # sys.exit(0)
    sensitivity = round(tp / (tp + fn), 2)
    # #print(sensitivity)
    # #sys.exit(0)
    specificity = round(tn / (tn + fp), 2)
    accuracy = round((tp + tn) / (tn + fp + fn + tp), 2)
    fpr, tpr, thresholds = metrics.roc_curve(fish_list, result_list, pos_label=1)
    auc = round(metrics.auc(fpr, tpr), 2)
    # #print(name , "TN:",tn, "FP:",fp, "FN:",fn, "TP:",tp, "Sensitivity:",sensitivity, "Specificity:",specificity, "AUC:",auc,"Accuracy:",accuracy)
    print(name, tn, fp, fn, tp, sensitivity, specificity, auc, accuracy)


input_file = sys.argv[1]
input_file = input_file.strip()
train_ori = []
train_pred = []
test_ori = []
test_pred = []
val_ori = []
val_pred = []
fobj = open(input_file)
dict_samp = {}
for line in fobj:
    line = line.strip()
    line = line.split(" ")
    line_list = line[1].split("_x_")
    # if line[3] == '1':
    #    line[3] = '0'
    # else:
    #    line[3] = '1'
    model_name = line[0]
    val = "1"
    if float(line[4]) > 0.5:
        val = "0"
    if line_list[0] + "__" + line[2] in dict_samp:
        dict_samp[line_list[0] + "__" + line[2]] = (
            dict_samp[line_list[0] + "__" + line[2]] + "__" + val
        )
    else:
        dict_samp[line_list[0] + "__" + line[2]] = val
    # if line[3] == "train":
    # train_ori.append(int(line[2]))
    # train_pred.append(val)
    # if line[3] == "test":
    # test_ori.append(int(line[2]))
    # test_pred.append(val)
    # if line[3] == "val":
    # val_ori.append(int(line[2]))
    # val_pred.append(val)
fobj.close()
for i in dict_samp:
    # print(i, dict_samp[i])
    # sys.exit(0)
    val = dict_samp[i].split("__")
    val = [int(i) for i in val]
    mean = np.mean(val)
    pred = 0
    if mean > 0.5:
        pred = 1
    print(i, mean, pred)
    sys.exit(0)
# report(test_ori, test_pred, model_name+" test")
# report(train_ori, train_pred, model_name+" train")
# report(val_ori, val_pred, model_name+" val")
