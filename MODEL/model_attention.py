import tensorflow as tf
import numpy as np

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

        return att_score, A, h, ins_labels, ins_logits_unnorm, ins_logits, \
               slide_score_unnorm, Y_prob, Y_hat, Y_true, predict_label