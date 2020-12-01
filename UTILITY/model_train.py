import tensorflow as tf
import sklearn
from sklearn import metrics
import os
import random
import statistics

from UTILITY.util import most_frequent, get_data_from_tf, load_optimizers, load_loss_func


def nb_optimize(img_features, slide_label, i_model, b_model, c_model, i_optimizer, b_optimizer, c_optimizer,
                i_loss_func, b_loss_func, n_class, c1, c2, mut_ex):

    with tf.GradientTape() as i_tape, tf.GradientTape() as b_tape, tf.GradientTape() as c_tape:

        att_score, A, h, ins_labels, ins_logits_unnorm, ins_logits, slide_score_unnorm, \
        Y_prob, Y_hat, Y_true, predict_slide_label = c_model.call(img_features, slide_label)

        ins_labels, ins_logits_unnorm, ins_logits = i_model.call(slide_label, h, A)
        ins_loss = list()
        for j in range(len(ins_logits)):
            i_loss = i_loss_func(tf.one_hot(ins_labels[j], 2), ins_logits[j])
            ins_loss.append(i_loss)
        if mut_ex:
            I_Loss = tf.math.add_n(ins_loss) / n_class
        else:
            I_Loss = tf.math.add_n(ins_loss)

        slide_score_unnorm, Y_hat, Y_prob, predict_slide_label, Y_true = b_model.call(slide_label, A, h)

        B_Loss = b_loss_func(Y_true, Y_prob)

        T_Loss = c1 * B_Loss + c2 * I_Loss

    i_grad = i_tape.gradient(I_Loss, i_model.trainable_weights)
    i_optimizer.apply_gradients(zip(i_grad, i_model.trainable_weights))

    b_grad = b_tape.gradient(B_Loss, b_model.trainable_weights)
    b_optimizer.apply_gradients(zip(b_grad, b_model.trainable_weights))

    c_grad = c_tape.gradient(T_Loss, c_model.trainable_weights)
    c_optimizer.apply_gradients(zip(c_grad, c_model.trainable_weights))

    return I_Loss, B_Loss, T_Loss, predict_slide_label


def b_optimize(batch_size, top_k_percent, n_samples, img_features, slide_label, i_model, b_model,
               c_model, i_optimizer, b_optimizer, c_optimizer, i_loss_func, b_loss_func,
               n_class, c1, c2, mut_ex):

    step_size = 0

    Ins_Loss = list()
    Bag_Loss = list()
    Total_Loss = list()

    label_predict = list()

    n_ins = top_k_percent * n_samples
    n_ins = int(n_ins)

    for n_step in range(0, (n_samples // batch_size + 1)):
        if step_size < (n_samples - batch_size):
            with tf.GradientTape() as i_tape, tf.GradientTape() as b_tape, tf.GradientTape() as c_tape:
                att_score, A, h, ins_labels, ins_logits_unnorm, ins_logits, slide_score_unnorm, \
                Y_prob, Y_hat, Y_true, predict_label = c_model.call(
                    img_features=img_features[step_size:(step_size + batch_size)],
                    slide_label=slide_label)

                ins_labels, ins_logits_unnorm, ins_logits = i_model.call(slide_label, h, A)

                ins_loss = list()
                for j in range(len(ins_logits)):
                    i_loss = i_loss_func(tf.one_hot(ins_labels[j], 2), ins_logits[j])
                    ins_loss.append(i_loss)
                if mut_ex:
                    Loss_I = tf.math.add_n(ins_loss) / n_class
                else:
                    Loss_I = tf.math.add_n(ins_loss)

                slide_score_unnorm, Y_hat, Y_prob, predict_label, Y_true = b_model.call(slide_label, A, h)

                Loss_B = b_loss_func(Y_true, Y_prob)

                Loss_T = c1 * Loss_B + c2 * Loss_I

            i_grad = i_tape.gradient(Loss_I, i_model.trainable_weights)
            i_optimizer.apply_gradients(zip(i_grad, i_model.trainable_weights))

            b_grad = b_tape.gradient(Loss_B, b_model.trainable_weights)
            b_optimizer.apply_gradients(zip(b_grad, b_model.trainable_weights))

            c_grad = c_tape.gradient(Loss_T, c_model.trainable_weights)
            c_optimizer.apply_gradients(zip(c_grad, c_model.trainable_weights))

        else:
            with tf.GradientTape() as i_tape, tf.GradientTape() as b_tape, tf.GradientTape() as c_tape:
                att_score, A, h, ins_labels, ins_logits_unnorm, ins_logits, slide_score_unnorm, \
                Y_prob, Y_hat, Y_true, predict_label = c_model.call(img_features=img_features[(step_size - n_ins):],
                                                                    slide_label=slide_label)

                ins_labels, ins_logits_unnorm, ins_logits = i_model.call(slide_label, h, A)

                ins_loss = list()
                for j in range(len(ins_logits)):
                    i_loss = i_loss_func(tf.one_hot(ins_labels[j], 2), ins_logits[j])
                    ins_loss.append(i_loss)
                if mut_ex:
                    Loss_I = tf.math.add_n(ins_loss) / n_class
                else:
                    Loss_I = tf.math.add_n(ins_loss)

                slide_score_unnorm, Y_hat, Y_prob, predict_label, Y_true = b_model.call(slide_label, A, h)

                Loss_B = b_loss_func(Y_true, Y_prob)

                Loss_T = c1 * Loss_B + c2 * Loss_I

            i_grad = i_tape.gradient(Loss_I, i_model.trainable_weights)
            i_optimizer.apply_gradients(zip(i_grad, i_model.trainable_weights))

            b_grad = b_tape.gradient(Loss_B, b_model.trainable_weights)
            b_optimizer.apply_gradients(zip(b_grad, b_model.trainable_weights))

            c_grad = c_tape.gradient(Loss_T, c_model.trainable_weights)
            c_optimizer.apply_gradients(zip(c_grad, c_model.trainable_weights))

        Ins_Loss.append(float(Loss_I))
        Bag_Loss.append(float(Loss_B))
        Total_Loss.append(float(Loss_T))

        label_predict.append(predict_label)

        step_size += batch_size

    I_Loss = statistics.mean(Ins_Loss)
    B_Loss = statistics.mean(Bag_Loss)
    T_Loss = statistics.mean(Total_Loss)

    predict_slide_label = most_frequent(label_predict)

    return I_Loss, B_Loss, T_Loss, predict_slide_label

def train_step(i_model, b_model, c_model, train_path, imf_norm_op,
               i_wd_op_name, b_wd_op_name, c_wd_op_name,
               i_optimizer_name, b_optimizer_name, c_optimizer_name,
               i_loss_name, b_loss_name, mut_ex, n_class, c1, c2,
               i_learn_rate, b_learn_rate, c_learn_rate,
               i_l2_decay, b_l2_decay, c_l2_decay,
               top_k_percent, batch_size, batch_op):

    i_optimizer, b_optimizer, c_optimizer = load_optimizers(i_wd_op_name=i_wd_op_name,
                                                            b_wd_op_name=b_wd_op_name,
                                                            c_wd_op_name=c_wd_op_name,
                                                            i_optimizer_name=i_optimizer_name,
                                                            b_optimizer_name=b_optimizer_name,
                                                            c_optimizer_name=c_optimizer_name,
                                                            i_learn_rate=i_learn_rate,
                                                            b_learn_rate=b_learn_rate,
                                                            c_learn_rate=c_learn_rate,
                                                            i_l2_decay=i_l2_decay,
                                                            b_l2_decay=b_l2_decay,
                                                            c_l2_decay=c_l2_decay)

    i_loss_func, b_loss_func = load_loss_func(i_loss_func_name=i_loss_name,
                                              b_loss_func_name=b_loss_name)

    loss_total = list()
    loss_ins = list()
    loss_bag = list()

    slide_true_label = list()
    slide_predict_label = list()

    train_sample_list = os.listdir(train_path)
    train_sample_list = random.sample(train_sample_list, len(train_sample_list))
    for i in train_sample_list:
        print('=', end="")
        single_train_data = train_path + i
        img_features, slide_label = get_data_from_tf(tf_path=single_train_data, imf_norm_op=imf_norm_op)
        # shuffle the order of img features list in order to reduce the side effects of randomly drop potential
        # number of patches' feature vectors during training when enable batch training option
        img_features = random.sample(img_features, len(img_features))

        if batch_op:
            I_Loss, B_Loss, T_Loss, predict_slide_label = b_optimize(batch_size=batch_size,
                                                                     top_k_percent=top_k_percent,
                                                                     n_samples=len(img_features),
                                                                     img_features=img_features,
                                                                     slide_label=slide_label,
                                                                     i_model=i_model,
                                                                     b_model=b_model,
                                                                     c_model=c_model,
                                                                     i_optimizer=i_optimizer,
                                                                     b_optimizer=b_optimizer,
                                                                     c_optimizer=c_optimizer,
                                                                     i_loss_func=i_loss_func,
                                                                     b_loss_func=b_loss_func,
                                                                     n_class=n_class,
                                                                     c1=c1, c2=c2, mut_ex=mut_ex)
        else:
            I_Loss, B_Loss, T_Loss, predict_slide_label = nb_optimize(img_features=img_features,
                                                                      slide_label=slide_label,
                                                                      i_model=i_model,
                                                                      b_model=b_model,
                                                                      c_model=c_model,
                                                                      i_optimizer=i_optimizer,
                                                                      b_optimizer=b_optimizer,
                                                                      c_optimizer=c_optimizer,
                                                                      i_loss_func=i_loss_func,
                                                                      b_loss_func=b_loss_func,
                                                                      n_class=n_class,
                                                                      c1=c1, c2=c2, mut_ex=mut_ex)

        loss_total.append(float(T_Loss))
        loss_ins.append(float(I_Loss))
        loss_bag.append(float(B_Loss))

        slide_true_label.append(slide_label)
        slide_predict_label.append(predict_slide_label)

    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(slide_true_label, slide_predict_label).ravel()
    train_tn = int(tn)
    train_fp = int(fp)
    train_fn = int(fn)
    train_tp = int(tp)

    train_sensitivity = round(train_tp / (train_tp + train_fn), 2)
    train_specificity = round(train_tn / (train_tn + train_fp), 2)
    train_acc = round((train_tp + train_tn) / (train_tn + train_fp + train_fn + train_tp), 2)

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(slide_true_label, slide_predict_label, pos_label=1)
    train_auc = round(sklearn.metrics.auc(fpr, tpr), 2)

    train_loss = statistics.mean(loss_total)
    train_ins_loss = statistics.mean(loss_ins)
    train_bag_loss = statistics.mean(loss_bag)

    return train_loss, train_ins_loss, train_bag_loss, train_tn, train_fp, train_fn, train_tp, train_sensitivity, \
           train_specificity, train_acc, train_auc