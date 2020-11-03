import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import PIL
import datetime
import os
import random
import shutil
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

        logits_in = list()

        for i in range(self.n_class * self.n_ins):
            logit_in = tf.math.softmax(ins_classifier(ins_in[i]))
            logits_in.append(logit_in)

        return ins_label_in, logits_in

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

        logits_out = list()

        for i in range(self.n_ins):
            logit_out = tf.math.softmax(ins_classifier(top_pos[i]))
            logits_out.append(logit_out)

        return ins_label_out, logits_out

    def call(self, bag_label, h, A):
        for i in range(self.n_class):
            ins_classifier = self.ins_classifier()[i]
            if i == bag_label:
                A_I = list()
                for j in range(len(A)):
                    a_i = A[j][0][i]
                    A_I.append(a_i)
                ins_label_in, logits_in = self.in_call(ins_classifier, h, A_I)
            else:
                if self.mut_ex:
                    A_O = list()
                    for j in range(len(A)):
                        a_o = A[j][0][i]
                        A_O.append(a_o)
                    ins_label_out, logits_out = self.out_call(ins_classifier, h, A_O)
                else:
                    continue

        if self.mut_ex:
            ins_labels = tf.concat(values=[ins_label_in, ins_label_out], axis=0)
            ins_logits = logits_in + logits_out
        else:
            ins_labels = ins_label_in
            ins_logits = logits_in

        return ins_labels, ins_logits


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
        Y_hat = tf.math.top_k(tf.reshape(slide_score_unnorm, (1, self.n_class)), 1)[1][-1]
        Y_prob = tf.math.softmax(
            tf.reshape(slide_score_unnorm, (1, self.n_class)))  # shape be (1,2), predictions for each of the classes

        Y_true = tf.one_hot([bag_label], 2)

        return slide_score_unnorm, Y_hat, Y_prob, Y_true


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
            ins_labels, ins_logits = self.ins_net.call(slide_label, h, A)

        slide_score_unnorm, Y_hat, Y_prob, Y_true = self.bag_net.call(slide_label, A, h)

        return att_score, A, h, ins_labels, ins_logits, slide_score_unnorm, Y_prob, Y_hat, Y_true
tfrecord = '/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/Data/CLAM/TFRECORD_ICIAR2018'
clam_dir = '/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/Data/CLAM'

def dataset_shuffle(dataset, path, percent=[0.8, 0.1, 0.1]):
    """
    Input Arg:
        dataset -> path where all tfrecord data stored
        path -> path where you want to save training, testing, and validation data folder
    """

    # return training, validation, and testing path name
    train = path + '/train'
    valid = path + '/valid'
    test = path + '/test'

    # create training, validation, and testing directory only if it is not existed
    if os.path.exists(train) == False:
        os.mkdir(os.path.join(clam_dir, 'train'))
    if os.path.exists(valid) == False:
        os.mkdir(os.path.join(clam_dir, 'valid'))
    if os.path.exists(test) == False:
        os.mkdir(os.path.join(clam_dir, 'test'))

    total_num_data = len(os.listdir(dataset))

    # only shuffle the data when train, validation, and test directory are all empty
    if len(os.listdir(train)) == 0 & len(os.listdir(valid)) == 0 & len(os.listdir(test)) == 0:
        train_names = random.sample(os.listdir(dataset), int(total_num_data * percent[0]))
        for i in train_names:
            train_srcpath = os.path.join(dataset, i)
            shutil.copy(train_srcpath, train)

        valid_names = random.sample(list(set(os.listdir(dataset)) - set(os.listdir(train))),
                                    int(total_num_data * percent[1]))
        for j in valid_names:
            valid_srcpath = os.path.join(dataset, j)
            shutil.copy(valid_srcpath, valid)

        test_names = list(set(os.listdir(dataset)) - set(os.listdir(train)) - set(os.listdir(valid)))
        for k in test_names:
            test_srcpath = os.path.join(dataset, k)
            shutil.copy(test_srcpath, test)

dataset_shuffle(tfrecord, clam_dir, percent=[0.8,0.1,0.1])

train_data = '/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/Data/CLAM/train/'
val_data = '/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/Data/CLAM/valid/'
test_data = '/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/Data/CLAM/test/'

def tf_shut_up(no_warn_op=False):
    if no_warn_op:
        tf.get_logger().setLevel('ERROR')
    else:
        print('Are you sure you want to receive the annoying TensorFlow Warning Messages?', \
              '\n', 'If not, check the value of your input prameter for this function and re-run it.')


def lom_func():
    losses = {
        'Hinge': tf.keras.losses.Hinge,
        'hinge': tf.keras.losses.hinge,
        'SquaredHinge': tf.keras.losses.SquaredHinge,
        'squaredhinge': tf.keras.losses.squared_hinge,
        'CategoricalHinge': tf.keras.losses.CategoricalHinge,
        'categoricalhinge': tf.keras.losses.categorical_hinge,
        'BinaryCrossentropy': tf.keras.losses.BinaryCrossentropy,
        'binarycrossentropy': tf.keras.losses.binary_crossentropy,
        'CategoricalCrossentropy': tf.keras.losses.CategoricalCrossentropy,
        'categoricalcrossentropy': tf.keras.losses.categorical_crossentropy
    }

    metrics = {
        'Accuracy': tf.keras.metrics.Accuracy(),
        'BinaryAccuracy': tf.keras.metrics.BinaryAccuracy(),
        'CategoricalAccuracy': tf.keras.metrics.CategoricalAccuracy(),
        'Precision': tf.keras.metrics.Precision(),
        'Recall': tf.keras.metrics.Recall(),
        'AUC': tf.keras.metrics.AUC(),
        'TP': tf.keras.metrics.TruePositives(),
        'FP': tf.keras.metrics.FalsePositives(),
        'TN': tf.keras.metrics.TrueNegatives(),
        'FN': tf.keras.metrics.FalseNegatives(),
        'Mean': tf.keras.metrics.Mean()
    }

    optimizers = {
        'Adam': tf.keras.optimizers.Adam,
        'Adamx': tf.keras.optimizers.Adamax,
        'AdamW': tfa.optimizers.AdamW
    }

    return losses, metrics, optimizers

losses, metrics, optimizers = lom_func()


def train_step(i_model, b_model, c_model, train_path, i_loss_func, b_loss_func, mutual_ex=False,
               n_class=2, c1=0.7, c2=0.3, learn_rate=2e-04, l2_decay=1e-05):
    loss_total = list()
    loss_ins = list()
    loss_bag = list()
    acc = list()
    auc = list()
    precision = list()
    recall = list()

    c_optimizer = optimizers['AdamW'](learning_rate=learn_rate, weight_decay=l2_decay)
    i_optimizer = optimizers['AdamW'](learning_rate=learn_rate, weight_decay=l2_decay)
    b_optimizer = optimizers['AdamW'](learning_rate=learn_rate, weight_decay=l2_decay)

    for i in os.listdir(train_path):
        print('=', end="")
        single_train_data = train_path + i
        img_features, slide_label = get_data_from_tf(single_train_data)

        train_tp = 0
        train_fp = 0
        train_tn = 0
        train_fn = 0

        with tf.GradientTape() as i_tape, tf.GradientTape() as b_tape, tf.GradientTape() as c_tape:
            att_score, A, h, ins_labels, ins_logits, slide_score_unnorm, Y_prob, Y_hat, Y_true = c_model.call(
                img_features, slide_label)
            ins_labels, ins_logits = i_model.call(slide_label, h, A)
            slide_score_unnorm, Y_hat, Y_prob, Y_true = b_model.call(slide_label, A, h)

            ins_loss = list()
            for i in range(len(ins_logits)):
                i_loss = i_loss_func(tf.one_hot(ins_labels[i], 2), ins_logits[i])
                ins_loss.append(i_loss)
            if mutual_ex:
                I_Loss = tf.math.add_n(ins_loss) / n_class
            else:
                I_Loss = tf.math.add_n(ins_loss)
            B_Loss = b_loss_func(Y_true, Y_prob)
            T_Loss = c1 * B_Loss + c2 * I_Loss

        i_grad = i_tape.gradient(I_Loss, i_model.trainable_variables)
        i_optimizer.apply_gradients(zip(i_grad, i_model.trainable_variables))

        b_grad = b_tape.gradient(B_Loss, b_model.trainable_variables)

        b_optimizer.apply_gradients(zip(b_grad, b_model.trainable_variables))

        c_grad = c_tape.gradient(T_Loss, c_model.trainable_variables)
        c_optimizer.apply_gradients(zip(c_grad, c_model.trainable_variables))

        loss_total.append(T_Loss)
        loss_ins.append(I_Loss)
        loss_bag.append(B_Loss)

        tp = metrics['TP'](Y_true, Y_prob)
        fp = metrics['FP'](Y_true, Y_prob)
        tn = metrics['TN'](Y_true, Y_prob)
        fn = metrics['FN'](Y_true, Y_prob)

        train_tp += tp
        train_fp += fp
        train_tn += tn
        train_fn += fn

        acc_value = metrics['BinaryAccuracy'](Y_true, Y_prob)
        auc_value = metrics['AUC'](Y_true, Y_prob)
        precision_value = metrics['Precision'](Y_true, Y_prob)
        recall_value = metrics['Recall'](Y_true, Y_prob)

        acc.append(acc_value)
        auc.append(auc_value)
        precision.append(precision_value)
        recall.append(recall_value)

    acc_train = tf.math.add_n(acc) / len(os.listdir(train_path))
    auc_train = tf.math.add_n(auc) / len(os.listdir(train_path))
    precision_train = tf.math.add_n(precision) / len(os.listdir(train_path))
    recall_train = tf.math.add_n(recall) / len(os.listdir(train_path))

    train_loss = tf.math.add_n(loss_total) / len(os.listdir(train_path))
    train_ins_loss = tf.math.add_n(loss_ins) / len(os.listdir(train_path))
    train_bag_loss = tf.math.add_n(loss_bag) / len(os.listdir(train_path))

    return train_loss, train_ins_loss, train_bag_loss, acc_train, auc_train, train_tp, train_fp, train_tn, train_fn, precision_train, recall_train


def val_step(c_model, val_path, i_loss_func, b_loss_func, mutual_ex=False, n_class=2, c1=0.7, c2=0.3):
    loss_t = list()
    loss_i = list()
    loss_b = list()
    acc = list()
    auc = list()
    precision = list()
    recall = list()

    for j in os.listdir(val_path):
        print('=', end="")
        single_val_data = val_path + j
        img_features, slide_label = get_data_from_tf(single_val_data)

        val_tp = 0
        val_fp = 0
        val_tn = 0
        val_fn = 0

        att_score, A, h, ins_labels, ins_logits, slide_score_unnorm, Y_prob, Y_hat, Y_true = c_model.call(img_features,
                                                                                                          slide_label)

        ins_loss = list()
        for i in range(len(ins_logits)):
            i_loss = i_loss_func(tf.one_hot(ins_labels[i], 2), ins_logits[i])
            ins_loss.append(i_loss)
        if mutual_ex:
            I_Loss = tf.math.add_n(ins_loss) / n_class
        else:
            I_Loss = tf.math.add_n(ins_loss)

        B_Loss = b_loss_func(Y_true, Y_prob)
        T_Loss = c1 * B_Loss + c2 * I_Loss

        loss_t.append(T_Loss)
        loss_i.append(I_Loss)
        loss_b.append(B_Loss)

        tp = metrics['TP'](Y_true, Y_prob)
        fp = metrics['FP'](Y_true, Y_prob)
        tn = metrics['TN'](Y_true, Y_prob)
        fn = metrics['FN'](Y_true, Y_prob)

        val_tp += tp
        val_fp += fp
        val_tn += tn
        val_fn += fn

        acc_value = metrics['BinaryAccuracy'](Y_true, Y_prob)
        auc_value = metrics['AUC'](Y_true, Y_prob)
        precision_value = metrics['Precision'](Y_true, Y_prob)
        recall_value = metrics['Recall'](Y_true, Y_prob)

        acc.append(acc_value)
        auc.append(auc_value)
        precision.append(precision_value)
        recall.append(recall_value)

    val_loss = tf.math.add_n(loss_t) / len(os.listdir(val_path))
    val_ins_loss = tf.math.add_n(loss_i) / len(os.listdir(val_path))
    val_bag_loss = tf.math.add_n(loss_b) / len(os.listdir(val_path))

    val_acc = tf.math.add_n(acc) / len(os.listdir(val_path))
    val_auc = tf.math.add_n(auc) / len(os.listdir(val_path))
    val_precision = tf.math.add_n(precision) / len(os.listdir(val_path))
    val_recall = tf.math.add_n(recall) / len(os.listdir(val_path))

    return val_loss, val_ins_loss, val_bag_loss, val_acc, val_auc, val_tp, val_fp, val_tn, val_fn, val_precision, val_recall

ins = Ins(dim_compress_features=512, n_class=2, n_ins=8, mut_ex=True)

s_bag = S_Bag(dim_compress_features=512, n_class=2)

s_clam = S_CLAM(att_gate=True, net_size='big', n_ins=8, n_class=2, mut_ex=False,
            dropout=True, drop_rate=.25, mil_ins=True, att_only=False)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = '/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/Data/CLAM/log/' + current_time + '/train'
val_log_dir = '/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/Data/CLAM/log/' + current_time + '/val'

def train_eval(train_log, val_log, epochs=1):
    train_summary_writer = tf.summary.create_file_writer(train_log)
    val_summary_writer = tf.summary.create_file_writer(val_log)
    for epoch in range(epochs):
        # Training Step
        start_time = time.time()
        train_loss, train_ins_loss, train_bag_loss, acc_train, auc_train, train_tp, train_fp, \
        train_tn, train_fn, precision_train, recall_train = train_step(
            i_model=ins, b_model=s_bag, c_model=s_clam, train_path=train_data,
            i_loss_func=losses['binarycrossentropy'], b_loss_func=losses['binarycrossentropy'],
            mutual_ex=True, n_class=2, c1=0.7, c2=0.3, learn_rate=2e-04, l2_decay=1e-05
        )
        with train_summary_writer.as_default():
            tf.summary.scalar('Total Loss', float(train_loss), step=epoch)
            tf.summary.scalar('Instance Loss', float(train_ins_loss), step=epoch)
            tf.summary.scalar('Bag Loss', float(train_bag_loss), step=epoch)
            tf.summary.scalar('Accuracy', float(acc_train), step=epoch)
            tf.summary.scalar('AUC', float(auc_train), step=epoch)
            tf.summary.scalar('Precision', float(precision_train), step=epoch)
            tf.summary.scalar('Recall', float(recall_train), step=epoch)
            tf.summary.histogram('True Positive', int(train_tp), step=epoch)
            tf.summary.histogram('False Positive', int(train_fp), step=epoch)
            tf.summary.histogram('True Negative', int(train_tn), step=epoch)
            tf.summary.histogram('False Negative', int(train_fn), step=epoch)
        # Validation Step
        val_loss, val_ins_loss, val_bag_loss, val_acc, val_auc, val_tp, val_fp, val_tn, \
        val_fn, val_precision, val_recall = val_step(
            c_model=s_clam, val_path=val_data, i_loss_func=losses['binarycrossentropy'],
            b_loss_func=losses['binarycrossentropy'], mutual_ex=True, n_class=2, c1=0.7, c2=0.3
        )
        with val_summary_writer.as_default():
            tf.summary.scalar('Total Loss', float(val_loss), step=epoch)
            tf.summary.scalar('Instance Loss', float(val_ins_loss), step=epoch)
            tf.summary.scalar('Bag Loss', float(val_bag_loss), step=epoch)
            tf.summary.scalar('Accuracy', float(val_acc), step=epoch)
            tf.summary.scalar('AUC', float(val_auc), step=epoch)
            tf.summary.scalar('Precision', float(val_precision), step=epoch)
            tf.summary.scalar('Recall', float(val_recall), step=epoch)
            tf.summary.histogram('True Positive', int(val_tp), step=epoch)
            tf.summary.histogram('False Positive', int(val_fp), step=epoch)
            tf.summary.histogram('True Negative', int(val_tn), step=epoch)
            tf.summary.histogram('False Negative', int(val_fn), step=epoch)
        epoch_run_time = time.time() - start_time
        template = '\n Epoch {},  Train Loss: {}, Train Accuracy: {}, Val Loss: {}, Val Accuracy: {}, Epoch Running Time: {}'
        print(template.format(epoch + 1,
                              f"{float(train_loss):.8}",
                              f"{float(acc_train):.4%}",
                              f"{float(val_loss):.8}",
                              f"{float(val_acc):.4%}",
                              "--- %s mins ---" % int(epoch_run_time / 60)))

tf_shut_up(no_warn_op=True)
train_eval(train_log=train_log_dir, val_log=val_log_dir, epochs=200)