import tensorflow as tf
import sklearn
from sklearn import metrics
import os
import random
import statistics

from UTILITY.util import get_data_from_tf


def nb_val(img_features, slide_label, i_model, b_model, c_model,
           i_loss_func, b_loss_func, n_class, c1, c2, mutual_ex):

    att_score, A, h, ins_labels, ins_logits_unnorm, ins_logits, slide_score_unnorm, \
    Y_prob, Y_hat, Y_true, predict_slide_label = c_model.call(img_features, slide_label)

    ins_labels, ins_logits_unnorm, ins_logits = i_model.call(slide_label, h, A)

    ins_loss = list()
    for j in range(len(ins_logits)):
        i_loss = i_loss_func(tf.one_hot(ins_labels[j], 2), ins_logits[j])
        ins_loss.append(i_loss)
    if mutual_ex:
        I_Loss = tf.math.add_n(ins_loss) / n_class
    else:
        I_Loss = tf.math.add_n(ins_loss)

    slide_score_unnorm, Y_hat, Y_prob, predict_slide_label, Y_true = b_model.call(slide_label, A, h)

    B_Loss = b_loss_func(Y_true, Y_prob)

    T_Loss = c1 * B_Loss + c2 * I_Loss

    return I_Loss, B_Loss, T_Loss, predict_slide_label


def b_val(batch_size, n_ins, n_samples, img_features, slide_label, i_model, b_model,
          c_model, i_loss_func, b_loss_func, n_class, c1, c2, mutual_ex):
    step_size = 0

    Ins_Loss = list()
    Bag_Loss = list()
    Total_Loss = list()

    label_predict = list()

    for n_step in range(0, (n_samples // batch_size + 1)):
        if step_size < (n_samples - batch_size):
            att_score, A, h, ins_labels, ins_logits_unnorm, ins_logits, slide_score_unnorm, \
            Y_prob, Y_hat, Y_true, predict_label = c_model.call(
                img_features=img_features[step_size:(step_size + batch_size)],
                slide_label=slide_label)

            ins_labels, ins_logits_unnorm, ins_logits = i_model.call(slide_label, h, A)

            ins_loss = list()
            for j in range(len(ins_logits)):
                i_loss = i_loss_func(tf.one_hot(ins_labels[j], 2), ins_logits[j])
                ins_loss.append(i_loss)
            if mutual_ex:
                Loss_I = tf.math.add_n(ins_loss) / n_class
            else:
                Loss_I = tf.math.add_n(ins_loss)

            slide_score_unnorm, Y_hat, Y_prob, predict_label, Y_true = b_model.call(slide_label, A, h)

            Loss_B = b_loss_func(Y_true, Y_prob)
            Loss_T = c1 * Loss_B + c2 * Loss_I

        else:
            att_score, A, h, ins_labels, ins_logits_unnorm, ins_logits, slide_score_unnorm, \
            Y_prob, Y_hat, Y_true, predict_label = c_model.call(img_features=img_features[(step_size - n_ins):],
                                                                slide_label=slide_label)

            ins_labels, ins_logits_unnorm, ins_logits = i_model.call(slide_label, h, A)

            ins_loss = list()
            for j in range(len(ins_logits)):
                i_loss = i_loss_func(tf.one_hot(ins_labels[j], 2), ins_logits[j])
                ins_loss.append(i_loss)
            if mutual_ex:
                Loss_I = tf.math.add_n(ins_loss) / n_class
            else:
                Loss_I = tf.math.add_n(ins_loss)

            slide_score_unnorm, Y_hat, Y_prob, predict_label, Y_true = b_model.call(slide_label, A, h)

            Loss_B = b_loss_func(Y_true, Y_prob)

            Loss_T = c1 * Loss_B + c2 * Loss_I

        Ins_Loss.append(float(Loss_I))
        Bag_Loss.append(float(Loss_B))
        Total_Loss.append(float(Loss_T))

        label_predict.append(predict_label)

        step_size += batch_size

    I_Loss = statistics.mean(Ins_Loss)
    B_Loss = statistics.mean(Bag_Loss)
    T_Loss = statistics.mean(Total_Loss)

    predict_slide_label = statistics.mode(label_predict)

    return I_Loss, B_Loss, T_Loss, predict_slide_label


def val_step(i_model, b_model, c_model, val_path, i_loss_func, b_loss_func, mutual_ex,
             n_class, c1, c2, n_ins, batch_size, batch_op):
    loss_t = list()
    loss_i = list()
    loss_b = list()

    slide_true_label = list()
    slide_predict_label = list()

    val_sample_list = os.listdir(val_path)
    val_sample_list = random.sample(val_sample_list, len(val_sample_list))
    for i in val_sample_list:
        print('=', end="")
        single_val_data = val_path + i
        img_features, slide_label = get_data_from_tf(single_val_data)

        img_features = random.sample(img_features, len(img_features))  # follow the training loop, see details there

        if batch_op:
            I_Loss, B_Loss, T_Loss, predict_slide_label = b_val(batch_size=batch_size, n_ins=n_ins,
                                                                n_samples=len(img_features),
                                                                img_features=img_features,
                                                                slide_label=slide_label,
                                                                i_model=i_model,
                                                                b_model=b_model,
                                                                c_model=c_model,
                                                                i_loss_func=i_loss_func,
                                                                b_loss_func=b_loss_func,
                                                                n_class=n_class, c1=c1, c2=c2,
                                                                mutual_ex=mutual_ex)
        else:
            I_Loss, B_Loss, T_Loss, predict_slide_label = nb_val(img_features=img_features,
                                                                 slide_label=slide_label,
                                                                 i_model=i_model,
                                                                 b_model=b_model,
                                                                 c_model=c_model,
                                                                 i_loss_func=i_loss_func,
                                                                 b_loss_func=b_loss_func,
                                                                 n_class=n_class, c1=c1, c2=c2,
                                                                 mutual_ex=mutual_ex)

        loss_t.append(float(T_Loss))
        loss_i.append(float(I_Loss))
        loss_b.append(float(B_Loss))

        slide_true_label.append(slide_label)
        slide_predict_label.append(predict_slide_label)

    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(slide_true_label, slide_predict_label).ravel()
    val_tn = int(tn)
    val_fp = int(fp)
    val_fn = int(fn)
    val_tp = int(tp)

    val_sensitivity = round(val_tp / (val_tp + val_fn), 2)
    val_specificity = round(val_tn / (val_tn + val_fp), 2)
    val_acc = round((val_tp + val_tn) / (val_tn + val_fp + val_fn + val_tp), 2)

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(slide_true_label, slide_predict_label, pos_label=1)
    val_auc = round(sklearn.metrics.auc(fpr, tpr), 2)

    val_loss = statistics.mean(loss_t)
    val_ins_loss = statistics.mean(loss_i)
    val_bag_loss = statistics.mean(loss_b)

    return val_loss, val_ins_loss, val_bag_loss, val_tn, val_fp, val_fn, val_tp, val_sensitivity, val_specificity, \
           val_acc, val_auc