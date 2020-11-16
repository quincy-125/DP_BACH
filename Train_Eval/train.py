import tensorflow as tf
import tensorflow_addons as tfa
from tqdm import tqdm
import sklearn
from sklearn import metrics
import numpy as np
import pickle as pkl
import PIL
import datetime
import os
import random
import shutil
import statistics
import time


def get_data_from_tf(tf_path):
    feature = {'height': tf.io.FixedLenFeature([], tf.int64),
               'width': tf.io.FixedLenFeature([], tf.int64),
               'depth': tf.io.FixedLenFeature([], tf.int64),
               'label': tf.io.FixedLenFeature([], tf.int64),
               'image/format': tf.io.FixedLenFeature([], tf.string),
               'image_name': tf.io.FixedLenFeature([], tf.string),
               'image/encoded': tf.io.FixedLenFeature([], tf.string),
               'image_feature': tf.io.FixedLenFeature([], tf.string)}

    tfrecord_dataset = tf.data.TFRecordDataset(tf_path)

    def _parse_image_function(key):
        return tf.io.parse_single_example(key, feature)

    CLAM_dataset = tfrecord_dataset.map(_parse_image_function)

    image_features = list()

    for tfrecord_value in CLAM_dataset:
        img_feature = tf.io.parse_tensor(tfrecord_value['image_feature'], 'float32')
        slide_labels = tfrecord_value['label']
        slide_label = int(slide_labels)
        image_features.append(img_feature)

    return image_features, slide_label


class NG_Att_Net(tf.keras.Model):
    def __init__(self, dim_features=1024, dim_compress_features=512, n_hidden_units=256, n_classes=2,
                 dropout=False, dropout_rate=.25):
        super(NG_Att_Net, self).__init__()
        self.dim_features = dim_features
        self.dim_compress_features = dim_compress_features
        self.n_hidden_units = n_hidden_units
        self.n_classes = n_classes
        self.dropout = dropout
        self.dropout_rate = dropout_rate

        self.compression_model = tf.keras.models.Sequential()
        self.model = tf.keras.models.Sequential()

        self.fc_compress_layer = tf.keras.layers.Dense(units=dim_compress_features, activation='relu',
                                                       input_shape=(dim_features,), kernel_initializer='glorot_normal',
                                                       bias_initializer='zeros', name='Fully_Connected_Layer')

        self.compression_model.add(self.fc_compress_layer)
        self.model.add(self.fc_compress_layer)

        self.att_layer1 = tf.keras.layers.Dense(units=n_hidden_units, activation='tanh',
                                                input_shape=(dim_compress_features,),
                                                kernel_initializer='glorot_normal', bias_initializer='zeros',
                                                name='Attention_Layer1')

        self.att_layer2 = tf.keras.layers.Dense(units=n_classes, activation='linear', input_shape=(n_hidden_units,),
                                                kernel_initializer='glorot_normal', bias_initializer='zeros',
                                                name='Attention_Layer2')

        self.model.add(self.att_layer1)

        if dropout:
            self.model.add(tf.keras.layers.Dropout(dropout_rate, name='Dropout_Layer'))

        self.model.add(self.att_layer2)

    def att_compress_model_no_gate(self):
        return self.compression_model

    def att_model_no_gate(self):
        return self.model

    def call(self, x):
        h = list()
        A = list()

        for i in x:
            c_imf = self.compression_model(i)
            h.append(c_imf)

        for i in x:
            a = self.model(i)
            A.append(a)
        return h, A


class G_Att_Net(tf.keras.Model):
    def __init__(self, dim_features=1024, dim_compress_features=512, n_hidden_units=256, n_classes=2,
                 dropout=False, dropout_rate=.25):
        super(G_Att_Net, self).__init__()
        self.dim_features = dim_features
        self.dim_compress_features = dim_compress_features
        self.n_hidden_units = n_hidden_units
        self.n_classes = n_classes
        self.dropout = dropout
        self.dropout_rate = dropout_rate

        self.compression_model = tf.keras.models.Sequential()
        self.model1 = tf.keras.models.Sequential()
        self.model2 = tf.keras.models.Sequential()
        self.model = tf.keras.models.Sequential()

        self.fc_compress_layer = tf.keras.layers.Dense(units=dim_compress_features, activation='relu',
                                                       input_shape=(dim_features,), kernel_initializer='glorot_normal',
                                                       bias_initializer='zeros', name='Fully_Connected_Layer')

        self.compression_model.add(self.fc_compress_layer)
        self.model1.add(self.fc_compress_layer)
        self.model2.add(self.fc_compress_layer)

        self.att_layer1 = tf.keras.layers.Dense(units=n_hidden_units, activation='tanh', input_shape=(dim_features,),
                                                kernel_initializer='glorot_normal', bias_initializer='zeros',
                                                name='Attention_Layer1')

        self.att_layer2 = tf.keras.layers.Dense(units=n_hidden_units, activation='sigmoid', input_shape=(dim_features,),
                                                kernel_initializer='glorot_normal', bias_initializer='zeros',
                                                name='Attention_Layer2')

        self.att_layer3 = tf.keras.layers.Dense(units=n_classes, activation='linear', input_shape=(n_hidden_units,),
                                                kernel_initializer='glorot_normal', bias_initializer='zeros',
                                                name='Attention_Layer3')

        self.model1.add(self.att_layer1)
        self.model2.add(self.att_layer2)

        if dropout:
            self.model1.add(tf.keras.layers.Dropout(dropout_rate, name='Dropout_Layer'))
            self.model2.add(tf.keras.layers.Dropout(dropout_rate, name='Dropout_Layer'))

        self.model.add(self.att_layer3)

    def att_compress_model_gate(self):
        return self.compression_model

    def att_model_gate(self):
        gated_att_net_list = [self.model1, self.model2, self.model]
        return gated_att_net_list

    def call(self, x):
        h = list()
        A = list()

        for i in x:
            c_imf = self.compression_model(i)
            h.append(c_imf)

        for i in x:
            layer1_output = self.model1(i)
            layer2_output = self.model2(i)
            a = tf.math.multiply(layer1_output, layer2_output)
            a = self.model(a)
            A.append(a)

        return h, A


class Ins(tf.keras.Model):
    def __init__(self, dim_compress_features=512, n_class=2, n_ins=8, mut_ex=False):
        super(Ins, self).__init__()
        self.dim_compress_features = dim_compress_features
        self.n_class = n_class
        self.n_ins = n_ins
        self.mut_ex = mut_ex

        self.ins_model = list()
        self.m_ins_model = tf.keras.models.Sequential()
        self.m_ins_layer = tf.keras.layers.Dense(
            units=self.n_class, activation='linear', input_shape=(self.dim_compress_features,),
            name='Instance_Classifier_Layer'
        )
        self.m_ins_model.add(self.m_ins_layer)

        for i in range(self.n_class):
            self.ins_model.append(self.m_ins_model)

    def ins_classifier(self):
        return self.ins_model

    @staticmethod
    def generate_pos_labels(n_pos_sample):
        return tf.fill(dims=[n_pos_sample, ], value=1)

    @staticmethod
    def generate_neg_labels(n_neg_sample):
        return tf.fill(dims=[n_neg_sample, ], value=0)

    def in_call(self, ins_classifier, h, A_I):
        pos_label = self.generate_pos_labels(self.n_ins)
        neg_label = self.generate_neg_labels(self.n_ins)
        ins_label_in = tf.concat(values=[pos_label, neg_label], axis=0)
        A_I = tf.reshape(tf.convert_to_tensor(A_I), (1, len(A_I)))

        top_pos_ids = tf.math.top_k(A_I, self.n_ins)[1][-1]
        pos_index = list()
        for i in top_pos_ids:
            pos_index.append(i)

        pos_index = tf.convert_to_tensor(pos_index)
        top_pos = list()
        for i in pos_index:
            top_pos.append(h[i])

        top_neg_ids = tf.math.top_k(-A_I, self.n_ins)[1][-1]
        neg_index = list()
        for i in top_neg_ids:
            neg_index.append(i)

        neg_index = tf.convert_to_tensor(neg_index)
        top_neg = list()
        for i in neg_index:
            top_neg.append(h[i])

        ins_in = tf.concat(values=[top_pos, top_neg], axis=0)
        logits_unnorm_in = list()
        logits_in = list()

        for i in range(self.n_class * self.n_ins):
            ins_score_unnorm_in = ins_classifier(ins_in[i])
            logit_in = tf.math.softmax(ins_score_unnorm_in)
            logits_unnorm_in.append(ins_score_unnorm_in)
            logits_in.append(logit_in)

        return ins_label_in, logits_unnorm_in, logits_in

    def out_call(self, ins_classifier, h, A_O):
        # get compressed 512-dimensional instance-level feature vectors for following use, denoted by h
        A_O = tf.reshape(tf.convert_to_tensor(A_O), (1, len(A_O)))
        top_pos_ids = tf.math.top_k(A_O, self.n_ins)[1][-1]
        pos_index = list()
        for i in top_pos_ids:
            pos_index.append(i)

        pos_index = tf.convert_to_tensor(pos_index)
        top_pos = list()
        for i in pos_index:
            top_pos.append(h[i])

        # mutually-exclusive -> top k instances w/ highest attention scores ==> false pos = neg
        pos_ins_labels_out = self.generate_neg_labels(self.n_ins)
        ins_label_out = pos_ins_labels_out

        logits_unnorm_out = list()
        logits_out = list()

        for i in range(self.n_ins):
            ins_score_unnorm_out = ins_classifier(top_pos[i])
            logit_out = tf.math.softmax(ins_score_unnorm_out)
            logits_unnorm_out.append(ins_score_unnorm_out)
            logits_out.append(logit_out)

        return ins_label_out, logits_unnorm_out, logits_out

    def call(self, bag_label, h, A):
        for i in range(self.n_class):
            ins_classifier = self.ins_classifier()[i]
            if i == bag_label:
                A_I = list()
                for j in range(len(A)):
                    a_i = A[j][0][i]
                    A_I.append(a_i)
                ins_label_in, logits_unnorm_in, logits_in = self.in_call(ins_classifier, h, A_I)
            else:
                if self.mut_ex:
                    A_O = list()
                    for j in range(len(A)):
                        a_o = A[j][0][i]
                        A_O.append(a_o)
                    ins_label_out, logits_unnorm_out, logits_out = self.out_call(ins_classifier, h, A_O)
                else:
                    continue

        if self.mut_ex:
            ins_labels = tf.concat(values=[ins_label_in, ins_label_out], axis=0)
            ins_logits_unnorm = logits_unnorm_in + logits_unnorm_out
            ins_logits = logits_in + logits_out
        else:
            ins_labels = ins_label_in
            ins_logits_unnorm = logits_unnorm_in
            ins_logits = logits_in

        return ins_labels, ins_logits_unnorm, ins_logits


class S_Bag(tf.keras.Model):
    def __init__(self, dim_compress_features=512, n_class=2):
        super(S_Bag, self).__init__()
        self.dim_compress_features = dim_compress_features
        self.n_class = n_class

        self.s_bag_model = tf.keras.models.Sequential()
        self.s_bag_layer = tf.keras.layers.Dense(
            units=1, activation='linear', input_shape=(self.n_class, self.dim_compress_features),
            name='Bag_Classifier_Layer'
        )
        self.s_bag_model.add(self.s_bag_layer)

    def bag_classifier(self):
        return self.s_bag_model

    def h_slide(self, A, h):
        # compute the slide-level representation aggregated per the attention score distribution for the mth class
        SAR = list()
        for i in range(len(A)):
            sar = tf.linalg.matmul(tf.transpose(A[i]), h[i])  # shape be (2,512)
            SAR.append(sar)
        slide_agg_rep = tf.math.add_n(SAR)  # return h_[slide,m], shape be (2,512)

        return slide_agg_rep

    def call(self, bag_label, A, h):
        slide_agg_rep = self.h_slide(A, h)
        bag_classifier = self.bag_classifier()
        slide_score_unnorm = bag_classifier(slide_agg_rep)
        slide_score_unnorm = tf.reshape(slide_score_unnorm, (1, self.n_class))
        Y_hat = tf.math.top_k(slide_score_unnorm, 1)[1][-1]
        Y_prob = tf.math.softmax(
            tf.reshape(slide_score_unnorm, (1, self.n_class)))  # shape be (1,2), predictions for each of the classes
        predict_label = np.argmax(Y_prob.numpy())

        Y_true = tf.one_hot([bag_label], 2)

        return slide_score_unnorm, Y_hat, Y_prob, predict_label, Y_true


class S_CLAM(tf.keras.Model):
    def __init__(self, att_gate=False, net_size='small', n_ins=8, n_class=2, mut_ex=False,
                 dropout=False, drop_rate=.25, mil_ins=False, att_only=False):
        super(S_CLAM, self).__init__()
        self.att_gate = att_gate
        self.net_size = net_size
        self.n_ins = n_ins
        self.n_class = n_class
        self.mut_ex = mut_ex
        self.dropout = dropout
        self.drop_rate = drop_rate
        self.mil_ins = mil_ins
        self.att_only = att_only

        self.net_shape_dict = {
            'small': [1024, 512, 256],
            'big': [1024, 512, 384]
        }
        self.net_shape = self.net_shape_dict[self.net_size]

        if self.att_gate:
            self.att_net = G_Att_Net(dim_features=self.net_shape[0], dim_compress_features=self.net_shape[1],
                                     n_hidden_units=self.net_shape[2],
                                     n_classes=self.n_class, dropout=self.dropout, dropout_rate=self.drop_rate)
        else:
            self.att_net = NG_Att_Net(dim_features=self.net_shape[0], dim_compress_features=self.net_shape[1],
                                      n_hidden_units=self.net_shape[2],
                                      n_classes=self.n_class, dropout=self.dropout, dropout_rate=self.drop_rate)

        self.bag_net = S_Bag(dim_compress_features=self.net_shape[1], n_class=self.n_class)

        self.ins_net = Ins(dim_compress_features=self.net_shape[1], n_class=self.n_class, n_ins=self.n_ins,
                           mut_ex=self.mut_ex)

    def call(self, img_features, slide_label):
        """
        Args:
            img_features -> original 1024-dimensional instance-level feature vectors
            slide_label -> ground-truth slide label, could be 0 or 1 for binary classification
        """

        h, A = self.att_net.call(img_features)
        att_score = A  # output from attention network
        A = tf.math.softmax(A)  # softmax onattention scores

        if self.att_only:
            return att_score

        if self.mil_ins:
            ins_labels, ins_logits_unnorm, ins_logits = self.ins_net.call(slide_label, h, A)

        slide_score_unnorm, Y_hat, Y_prob, predict_label, Y_true = self.bag_net.call(slide_label, A, h)

        return att_score, A, h, ins_labels, ins_logits_unnorm, ins_logits, slide_score_unnorm, Y_prob, Y_hat, Y_true, predict_label


train_data = '/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/Data/CLAM/BACH/train/'
val_data = '/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/Data/CLAM/BACH/val/'
test_data = '/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/Data/CLAM/BACH/test/'


def tf_shut_up(no_warn_op=False):
    if no_warn_op:
        tf.get_logger().setLevel('ERROR')
    else:
        print('Are you sure you want to receive the annoying TensorFlow Warning Messages?', \
              '\n', 'If not, check the value of your input prameter for this function and re-run it.')


def train_step(i_model, b_model, c_model, train_path, i_optimizer_func, b_optimizer_func,
               c_optimizer_func, i_loss_func, b_loss_func, mutual_ex, n_class, c1, c2, learn_rate, l2_decay):
    loss_total = list()
    loss_ins = list()
    loss_bag = list()

    i_optimizer = i_optimizer_func(learning_rate=learn_rate, weight_decay=l2_decay)
    b_optimizer = b_optimizer_func(learning_rate=learn_rate, weight_decay=l2_decay)
    c_optimizer = c_optimizer_func(learning_rate=learn_rate, weight_decay=l2_decay)

    slide_true_label = list()
    slide_predict_label = list()

    train_sample_list = os.listdir(train_path)
    train_sample_list = random.sample(train_sample_list, len(train_sample_list))
    for i in train_sample_list:
        print('=', end="")
        single_train_data = train_path + i
        img_features, slide_label = get_data_from_tf(single_train_data)

        with tf.GradientTape() as i_tape, tf.GradientTape() as b_tape, tf.GradientTape() as c_tape:
            att_score, A, h, ins_labels, ins_logits_unnorm, ins_logits, slide_score_unnorm, \
            Y_prob, Y_hat, Y_true, predict_label = c_model.call(img_features, slide_label)

            ins_labels, ins_logits_unnorm, ins_logits = i_model.call(slide_label, h, A)
            ins_loss = list()
            for j in range(len(ins_logits)):
                i_loss = i_loss_func(tf.one_hot(ins_labels[j], 2), ins_logits[j])
                ins_loss.append(i_loss)
            if mutual_ex:
                I_Loss = tf.math.add_n(ins_loss) / n_class
            else:
                I_Loss = tf.math.add_n(ins_loss)

            slide_score_unnorm, Y_hat, Y_prob, predict_label, Y_true = b_model.call(slide_label, A, h)
            B_Loss = b_loss_func(Y_true, Y_prob)
            T_Loss = c1 * B_Loss + c2 * I_Loss

        i_grad = i_tape.gradient(I_Loss, i_model.trainable_weights)
        i_optimizer.apply_gradients(zip(i_grad, i_model.trainable_weights))

        b_grad = b_tape.gradient(B_Loss, b_model.trainable_weights)
        b_optimizer.apply_gradients(zip(b_grad, b_model.trainable_weights))

        c_grad = c_tape.gradient(T_Loss, c_model.trainable_weights)
        c_optimizer.apply_gradients(zip(c_grad, c_model.trainable_weights))

        loss_total.append(float(T_Loss))
        loss_ins.append(float(I_Loss))
        loss_bag.append(float(B_Loss))

        slide_true_label.append(slide_label)
        slide_predict_label.append(predict_label)

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


def val_step(i_model, b_model, c_model, val_path, i_loss_func, b_loss_func, mutual_ex, n_class, c1, c2):
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

        att_score, A, h, ins_labels, ins_logits_unnorm, ins_logits, slide_score_unnorm, \
        Y_prob, Y_hat, Y_true, predict_label = c_model.call(img_features, slide_label)

        ins_labels, ins_logits_unnorm, ins_logits = i_model.call(slide_label, h, A)

        ins_loss = list()
        for j in range(len(ins_logits)):
            i_loss = i_loss_func(tf.one_hot(ins_labels[j], 2), ins_logits[j])
            ins_loss.append(i_loss)
        if mutual_ex:
            I_Loss = tf.math.add_n(ins_loss) / n_class
        else:
            I_Loss = tf.math.add_n(ins_loss)

        slide_score_unnorm, Y_hat, Y_prob, predict_label, Y_true = b_model.call(slide_label, A, h)

        B_Loss = b_loss_func(Y_true, Y_prob)
        T_Loss = c1 * B_Loss + c2 * I_Loss

        loss_t.append(float(T_Loss))
        loss_i.append(float(I_Loss))
        loss_b.append(float(B_Loss))

        slide_true_label.append(slide_label)
        slide_predict_label.append(predict_label)

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


def test(i_model, b_model, c_model, test_path):
    start_time = time.time()

    slide_true_label = list()
    slide_predict_label = list()

    for k in os.listdir(test_path):
        print('=', end="")
        single_test_data = test_path + k
        img_features, slide_label = get_data_from_tf(single_test_data)

        att_score, A, h, ins_labels, ins_logits_unnorm, ins_logits, slide_score_unnorm, \
        Y_prob, Y_hat, Y_true, predict_label = c_model.call(img_features, slide_label)

        ins_labels, ins_logits_unnorm, ins_logits = i_model.call(slide_label, h, A)
        slide_score_unnorm, Y_hat, Y_prob, predict_label, Y_true = b_model.call(slide_label, A, h)

        slide_true_label.append(slide_label)
        slide_predict_label.append(predict_label)

        print(slide_label, '\t', predict_label, '\t', Y_prob)

    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(slide_true_label, slide_predict_label).ravel()
    test_tn = int(tn)
    test_fp = int(fp)
    test_fn = int(fn)
    test_tp = int(tp)

    test_sensitivity = round(test_tp / (test_tp + test_fn), 2)
    test_specificity = round(test_tn / (test_tn + test_fp), 2)
    test_acc = round((test_tp + test_tn) / (test_tn + test_fp + test_fn + test_tp), 2)

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(slide_true_label, slide_predict_label, pos_label=1)
    test_auc = round(sklearn.metrics.auc(fpr, tpr), 2)

    test_run_time = time.time() - start_time

    template = '\n Test Accuracy: {}, Test Sensitivity: {}, Test Specificity: {}, Test Running Time: {}'
    print(template.format(f"{float(test_acc):.4%}",
                          f"{float(test_sensitivity):.4%}",
                          f"{float(test_specificity):.4%}",
                          "--- %s mins ---" % int(test_run_time / 60)))

    return test_tn, test_fp, test_fn, test_tp, test_sensitivity, test_specificity, test_acc, test_auc


ins = Ins(dim_compress_features=512, n_class=2, n_ins=8, mut_ex=True)

s_bag = S_Bag(dim_compress_features=512, n_class=2)

s_clam = S_CLAM(att_gate=True, net_size='big', n_ins=8, n_class=2, mut_ex=False,
                dropout=True, drop_rate=.25, mil_ins=True, att_only=False)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = '/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/Data/CLAM/log/' + current_time + '/train'
val_log_dir = '/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/Data/CLAM/log/' + current_time + '/val'


def train_eval(train_log, val_log, train_path, val_path, i_model, b_model,
               c_model, i_optimizer_func, b_optimizer_func, c_optimizer_func, i_loss_func,
               b_loss_func, mutual_ex, n_class, c1, c2, learn_rate, l2_decay, epochs):
    train_summary_writer = tf.summary.create_file_writer(train_log)
    val_summary_writer = tf.summary.create_file_writer(val_log)

    for epoch in range(epochs):
        # Training Step
        start_time = time.time()

        train_loss, train_ins_loss, train_bag_loss, train_tn, train_fp, train_fn, train_tp, \
        train_sensitivity, train_specificity, train_acc, train_auc = train_step(
            i_model=i_model, b_model=b_model, c_model=c_model, train_path=train_path,
            i_optimizer_func=i_optimizer_func, b_optimizer_func=b_optimizer_func, c_optimizer_func=c_optimizer_func,
            i_loss_func=i_loss_func, b_loss_func=b_loss_func, mutual_ex=mutual_ex,
            n_class=n_class, c1=c1, c2=c2, learn_rate=learn_rate, l2_decay=l2_decay)

        with train_summary_writer.as_default():
            tf.summary.scalar('Total Loss', float(train_loss), step=epoch)
            tf.summary.scalar('Instance Loss', float(train_ins_loss), step=epoch)
            tf.summary.scalar('Bag Loss', float(train_bag_loss), step=epoch)
            tf.summary.scalar('Accuracy', float(train_acc), step=epoch)
            tf.summary.scalar('AUC', float(train_auc), step=epoch)
            tf.summary.scalar('Sensitivity', float(train_sensitivity), step=epoch)
            tf.summary.scalar('Specificity', float(train_specificity), step=epoch)
            tf.summary.histogram('True Positive', int(train_tp), step=epoch)
            tf.summary.histogram('False Positive', int(train_fp), step=epoch)
            tf.summary.histogram('True Negative', int(train_tn), step=epoch)
            tf.summary.histogram('False Negative', int(train_fn), step=epoch)

        # Validation Step
        val_loss, val_ins_loss, val_bag_loss, val_tn, val_fp, val_fn, val_tp, \
        val_sensitivity, val_specificity, val_acc, val_auc = val_step(
            i_model=i_model, b_model=b_model, c_model=c_model, val_path=val_path,
            i_loss_func=i_loss_func, b_loss_func=b_loss_func, mutual_ex=mutual_ex, n_class=n_class, c1=c1, c2=c2)

        with val_summary_writer.as_default():
            tf.summary.scalar('Total Loss', float(val_loss), step=epoch)
            tf.summary.scalar('Instance Loss', float(val_ins_loss), step=epoch)
            tf.summary.scalar('Bag Loss', float(val_bag_loss), step=epoch)
            tf.summary.scalar('Accuracy', float(val_acc), step=epoch)
            tf.summary.scalar('AUC', float(val_auc), step=epoch)
            tf.summary.scalar('Sensitivity', float(val_sensitivity), step=epoch)
            tf.summary.scalar('Specificity', float(val_specificity), step=epoch)
            tf.summary.histogram('True Positive', int(val_tp), step=epoch)
            tf.summary.histogram('False Positive', int(val_fp), step=epoch)
            tf.summary.histogram('True Negative', int(val_tn), step=epoch)
            tf.summary.histogram('False Negative', int(val_fn), step=epoch)

        epoch_run_time = time.time() - start_time
        template = '\n Epoch {},  Train Loss: {}, Train Accuracy: {}, Val Loss: {}, Val Accuracy: {}, Epoch Running ' \
                   'Time: {} '
        print(template.format(epoch + 1,
                              f"{float(train_loss):.8}",
                              f"{float(train_acc):.4%}",
                              f"{float(val_loss):.8}",
                              f"{float(val_acc):.4%}",
                              "--- %s mins ---" % int(epoch_run_time / 60)))


def clam_main(train_log, val_log, train_path, val_path, test_path,
              i_model, b_model, c_model, i_optimizer_func, b_optimizer_func,
              c_optimizer_func, i_loss_func, b_loss_func, mutual_ex,
              n_class, c1, c2, learn_rate, l2_decay, epochs):
    train_eval(train_log=train_log, val_log=val_log, train_path=train_path,
               val_path=val_path, i_model=i_model, b_model=b_model, c_model=c_model,
               i_optimizer_func=i_optimizer_func, b_optimizer_func=b_optimizer_func,
               c_optimizer_func=c_optimizer_func, i_loss_func=i_loss_func,
               b_loss_func=b_loss_func, mutual_ex=mutual_ex, n_class=n_class, c1=c1, c2=c2,
               learn_rate=learn_rate, l2_decay=l2_decay, epochs=epochs)

    test_tn, test_fp, test_fn, test_tp, test_sensitivity, test_specificity, \
    test_acc, test_auc = test(i_model=i_model, b_model=b_model, c_model=c_model, test_path=test_path)


tf_shut_up(no_warn_op=True)


clam_main(train_log=train_log_dir, val_log=val_log_dir, train_path=train_data,
          val_path=val_data, test_path=test_data, i_model=ins, b_model=s_bag, c_model=s_clam,
          i_optimizer_func=tfa.optimizers.AdamW, b_optimizer_func=tfa.optimizers.AdamW,
          c_optimizer_func=tfa.optimizers.AdamW, i_loss_func=tf.keras.losses.binary_crossentropy,
          b_loss_func=tf.keras.losses.binary_crossentropy, mutual_ex=True, n_class=2,
          c1=0.7, c2=0.3, learn_rate=2e-04, l2_decay=1e-05, epochs=200)
