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
    fpr, tpr, thresholds = metrics.roc_curve(fish_list, result_list, pos_label=1)
    auc = round(metrics.auc(fpr, tpr), 2)
    print(
        name,
        "TN:",
        tn,
        "FP:",
        fp,
        "FN:",
        fn,
        "TP:",
        tp,
        "Sensitivity:",
        sensitivity,
        "Specificity:",
        specificity,
        "AUC:",
        auc,
    )


dict_arr = {}
fobj = open("train.txt")
header = fobj.readline()
header = fobj.readline()
# header = header.strip()
train_label = []
train_label_argmax = []
train_softmax = []
val_label = []
val_softmax = []
val_label_argmax = []
test_label = []
test_softmax = []
test_label_argmax = []
for i in fobj:
    i = i.strip()
    list = i.split(" ")
    Category = list[0]
    Label = list[1]
    Softmax = list[2]
    if Category == "train" or Category == "valid":
        Label = int(Label)
        Softmax = float(Softmax)
        # Pred = int(Pred)
        if Category == "train":
            train_label.append(Label)
            train_softmax.append(Softmax)
            # train_label_argmax.append(Pred)
        if Category == "valid":
            val_label.append(Label)
            val_softmax.append(Softmax)
            # val_label_argmax.append(Pred)
        # if Category == "test":
        # test_label.append(Label)
        # test_softmax.append(Softmax)
        # test_label_argmax.append(Pred)
fobj.close()


# report(train_label, train_label_argmax, "train")
# report(val_label, val_label_argmax, "val")
# report(test_label, test_label_argmax, "test")
# sys.exit(0)
# getting percentiles from prob distribution
perc_prob = [np.percentile(train_softmax, i) for i in range(1, 100, 1)]
# finding the best threshold from the percentiles
prob_aucs = [
    roc_auc_score(train_label, [1 if x >= perc_prob[i] else 0 for x in train_softmax])
    for i in range(len(perc_prob))
]
for i in range(len(perc_prob)):
    print(perc_prob[i], prob_aucs[i])
# getting the index for best auc
idx_opti = np.argmax(prob_aucs)
# best threshold
threshold = perc_prob[idx_opti]
# threshold=0.5
print("Threshold", threshold)
# calculating training accuracy
opti_thres_pred = [1 if x >= threshold else 0 for x in train_softmax]
training_acc = roc_auc_score(train_label, opti_thres_pred)
# training_acc_argmax = roc_auc_score(train_label, train_label_argmax)
print("training_acc", training_acc)
# print("training_acc_argmax",training_acc_argmax)

# calculating validation accuracy
opti_thres_pred = [1 if x >= threshold else 0 for x in val_softmax]
validation_acc = roc_auc_score(val_label, opti_thres_pred)
# validation_acc_argmax = roc_auc_score(val_label, val_label_argmax)
print("validation_acc", validation_acc)
# print("validation_acc_argmax",validation_acc_argmax)

# #calculating testing accuracy
# opti_thres_pred=[ 1 if x >= threshold else 0 for x in test_softmax ]
# testing_acc = roc_auc_score(test_label, opti_thres_pred)
# testing_acc_argmax = roc_auc_score(test_label, test_label_argmax)
# print("testing_acc",testing_acc)
# print("testing_acc_argmax",testing_acc_argmax)
