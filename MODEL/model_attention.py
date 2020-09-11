## Create attention neural network with Keras

## Import required modules/packages

import numpy as np
import tensorflow as tf

"""
2 options for Attention Network
    1. Attention Network w/ Gating
    2. Attention Network w.o. Gating

>> Create new class of Attention Network which include the above 2 classes of attention network options
"""

"""
Read Image Feature Vectors [1024-dimensional] from tfrecord file
"""

# None-Gated Attention Network Class - assign the same weights of each attention head/layer
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

    def compress(self, x):
        h = list()
        for i in x:
            c_imf = self.compression_model(i)
            h.append(c_imf)
        return h

    def forward(self, x):
        A = list()
        for i in x:
            a = self.model(i)
            A.append(a)
        return A


# Gated Attention Network Class - scaling the weights of each attention head/layer -> weights of each attention layer
# would be different
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

        # GlorotNormal <==> Xavier for weights initialization
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

    def compress(self, x):
        h = list()
        for i in x:
            c_imf = self.compression_model(i)
            h.append(c_imf)
        return h

    def forward(self, x):
        A = list()
        for i in x:
            layer1_output = self.model1(i)  # output from the first dense layer
            layer2_output = self.model2(i)  # output from the second dense layer
            a = tf.math.multiply(layer1_output, layer2_output)  # cross product of the outputs from 1st and 2nd layer
            a = self.model(a)  # pass the output of the product of the outputs from 1st 2 layers to the last layer
            A.append(a)

        return A


# CLAM Class - Attention Network (Gated/None-Gated) + Instance-Level Clustering
class CLAM(tf.keras.Model):
    def __init__(self, att_net_gate=False, net_siz_arg='small', n_instance_sample=8, n_classes=2, subtype_prob=False,
                 dropout=False, dropout_rate=.25, mil_loss_func=tf.keras.losses.CategoricalCrossentropy()):
        super(CLAM, self).__init__()
        self.att_net_gate = att_net_gate
        self.net_size_arg = net_siz_arg
        self.n_instance_sample = n_instance_sample
        self.n_classes = n_classes
        self.subtype_prob = subtype_prob
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.mil_loss_func = mil_loss_func

        self.size_dictionary = {
            'small': [1024, 512, 256],
            'big': [1024, 512, 384]
        }
        size = self.size_dictionary[net_siz_arg]

        if att_net_gate:
            self.att_net = G_Att_Net(dim_features=size[0], dim_compress_features=size[1], n_hidden_units=size[2],
                                     n_classes=n_classes, dropout=dropout, dropout_rate=dropout_rate)
        else:
            self.att_net = NG_Att_Net(dim_features=size[0], dim_compress_features=size[1], n_hidden_units=size[2],
                                      n_classes=n_classes, dropout=dropout, dropout_rate=dropout_rate)

        # Multi-Instance Learning - Adding 2 classifier models, one for bag-level, one for instance-level
        # Bag-level classifier model
        self.bag_classifiers = list()  # list of keras sequential model w/ single linear dense layer for each class
        for i in range(n_classes):
            self.bag_classifier = tf.keras.models.Sequential(
                tf.keras.layers.Dense(units=1, activation='linear', input_shape=(size[1],))  # W_[c,m] shape be (1,512)
            )  # independent sequential model w/ single linear dense layer to do slide-level prediction for each class
            self.bag_classifiers.append(self.bag_classifier)

        # Instance-level classifier model
        # for each of n classes, take transpose of compressed img feature for kth patch (h_k) with shape (512,1) in,
        # and return the cluster assignment score predicted for kth patch (P_[m,k]) with shape (2,1)
        self.instance_classifiers = list()
        for i in range(n_classes):
            self.instance_classifier = tf.keras.models.Sequential(
                tf.keras.layers.Dense(units=self.n_classes, activation='linear', input_shape=(size[1],))
            )  # W_[inst,m] shape (2,512)
            self.instance_classifiers.append(self.instance_classifier)

    # Generate patch-level pseudo labels with staticmethod [default values -> 1 for positive, 0 for negative]
    # Generate positive patch-level pseudo labels
    @staticmethod
    def generate_pos_labels(n_pos_sample):
        return tf.fill(dims=[n_pos_sample, ], value=1.0)

    # Generate negative patch-level pseudo labels
    @staticmethod
    def generate_neg_labels(n_neg_sample):
        return tf.fill(dims=[n_neg_sample, ], value=0.0)

    # Self-defined function equivalent to torch.index_select() with staticmethod
    # Usage -> get top k pos/neg instances based on the generated indexes by sorting their attention scores
    @staticmethod
    def tf_index_select(input, dim, index):
        """
        input_(tensor): input tensor
        dim(int): dimension
        index (LongTensor)  the 1-D tensor containing the indices to index
        """
        shape = input.get_shape().as_list()
        if dim == -1:
            dim = len(shape) - 1
        shape[dim] = 1

        tmp = []
        for idx in index:
            begin = [0] * len(shape)
            begin[dim] = idx
            tmp.append(tf.slice(input, begin, shape))
        res = tf.concat(tmp, axis=dim)

        return res

    # Apply Multi-Instance Learning to perform in-class and out-class instance-level clustering
    # In-class attention branch based instance-level clustering
    def instance_clustering_in_class(self, A, h, classifier):
        pos_pseudo_labels = self.generate_pos_labels(self.n_instance_sample)
        neg_pseudo_labels = self.generate_neg_labels(self.n_instance_sample)
        pseudo_labels = tf.concat(values=[pos_pseudo_labels, neg_pseudo_labels], axis=0)
        A = tf.reshape(tf.convert_to_tensor(A), (1, len(A) * self.n_classes))

        top_pos_ids = tf.math.top_k(A, self.n_instance_sample)[1][-1]
        pos_index = list()
        for i in top_pos_ids:
            if i % 2 == 0:
                pos_index.append(i // 2)
            else:
                pos_index.append((i + 1) // 2)
        pos_index = tf.convert_to_tensor(pos_index)
        top_pos = list()
        for i in pos_index:
            # print(f'Pos index h is, {len(h)}, i is {i}')
            top_pos.append(h[i - 1])

        top_neg_ids = tf.math.top_k(-A, self.n_instance_sample)[1][-1]
        neg_index = list()
        for i in top_neg_ids:
            if i % 2 == 0:
                neg_index.append(i // 2)
            else:
                neg_index.append((i + 1) // 2)
        neg_index = tf.convert_to_tensor(neg_index)
        top_neg = list()
        for i in neg_index:
            # print('shape of h, ', h[i].shape, 'i is ', i)
            # shape of h,  (1, 512) i is  tf.Tensor(16, shape=(), dtype=int32)
            # print(f'Neg index h is, {len(h)}, i is {i}')
            top_neg.append(h[i - 1])

        instance_samples = tf.concat(values=[top_pos, top_neg], axis=0)

        logits = list()
        instance_loss = list()

        for i in range(self.n_instance_sample):
            logit = tf.reshape(classifier(instance_samples[i]), (2, 1))
            ins_loss = self.mil_loss_func(pseudo_labels[i], logit)
            logits.append(logit)
            instance_loss.append(ins_loss)

        instance_predict = tf.sort(logits, direction='ASCENDING')
        instance_predict = tf.reshape(tf.convert_to_tensor(instance_predict),
                                      (1, len(instance_predict) * self.n_classes))
        pos_predict = instance_predict[0][:self.n_instance_sample]
        neg_predict = instance_predict[0][self.n_instance_sample:]
        accurate_pos_predict = (pos_predict == tf.constant(pos_pseudo_labels)).numpy().tolist().count(True)
        accurate_neg_predict = (neg_predict == tf.constant(neg_pseudo_labels)).numpy().tolist().count(True)

        pos_acc = accurate_pos_predict / self.n_instance_sample
        neg_acc = accurate_neg_predict / self.n_instance_sample

        return instance_loss, pos_acc, neg_acc

    # Out-class attention branch based instance-level clustering [Optional Functionality]
    def instance_level_clustering_out_class(self, A, h, classifier):
        # get compressed 512-dimensional instance-level feature vectors for following use, denoted by h
        A = tf.reshape(tf.convert_to_tensor(A), (1, len(A) * self.n_classes))
        top_pos_ids = tf.math.top_k(A, self.n_instance_sample)[1][-1]
        pos_index = list()
        for i in top_pos_ids:
            if i % 2 == 0:
                pos_index.append(i // 2)
            else:
                pos_index.append((i + 1) // 2)
        pos_index = tf.convert_to_tensor(pos_index)
        top_pos = list()
        for i in pos_index:
            top_pos.append(h[i - 1])

        # mutually-exclusive -> top k instances w/ highest attention scores ==> false pos = neg
        pos_pseudo_labels = self.generate_neg_labels(self.n_instance_sample)
        logits = list()
        instance_loss = list()

        for i in range(self.n_instance_sample):
            logit = tf.reshape(classifier(top_pos[i]), (2, 1))
            ins_loss = self.mil_loss_func(pos_pseudo_labels[i], logit)
            logits.append(logit)
            instance_loss.append(ins_loss)

        pos_predict = tf.sort(logits, direction='ASCENDING')
        pos_predict = tf.reshape(tf.convert_to_tensor(pos_predict), (1, len(pos_predict) * self.n_classes))
        pos_predict = pos_predict[0][:self.n_instance_sample]
        accurate_pos_predict = (pos_predict == tf.constant(pos_pseudo_labels)).numpy().tolist().count(True)
        pos_acc = accurate_pos_predict / self.n_instance_sample

        # top k instances w/ lowest attention scores -> false neg != pos ==> excluded
        neg_acc = -1  # never pick top k neg instances in out-the-class instance-level clustering, set this be -1

        return instance_loss, pos_acc, neg_acc

    def forward(self, img_features, slide_label, mil_op=False, slide_predict_op=False, att_only_op=False):
        """
        Args:
            img_features -> original 1024-dimensional instance-level feature vectors
            labels -> mutable entire label set, could be 0 or 1 for binary classification
            mil_op -> whether or not perform the instance-level clustering, default be False
            slide_predict_op ->
            att_only_op -> if only return the attention scores, default be False
        """

        # get the compressed 512-dim feature vectors for following use
        h = self.att_net.compress(img_features)

        A = self.att_net.forward(img_features)
        att_net_out = A  # output from attention network
        A = tf.math.softmax(A)  # attention scores computed by Eqa#1 in CLAM paper

        if att_only_op:
            CLAM_outcomes = {
                'Attention_Scores': att_net_out
            }
            return CLAM_outcomes  # return attention scores of the kth patch for the mth class (i.e. a_[k,m])

        if mil_op:
            instance_loss_total = 0.0
            pos_acc_total = 0.0
            neg_acc_total = 0.0

            for i in range(len(self.instance_classifiers)):

                classifier = self.instance_classifiers[i]
                if i == slide_label:
                    instance_loss, pos_acc, neg_acc = self.instance_clustering_in_class(A, h, classifier)

                    pos_acc_total += pos_acc
                    neg_acc_total += neg_acc
                else:
                    if self.subtype_prob:  # classes are mutually-exclusive assumption holds
                        instance_loss, pos_acc, neg_acc = self.instance_level_clustering_out_class(A, h, classifier)
                        pos_acc += pos_acc
                    else:  # classes are mutually-exclusive assumption not holds
                        continue
                instance_loss_total = sum(instance_loss)

            if self.subtype_prob:
                pos_acc_total /= len(self.instance_classifiers)
                instance_loss_total /= len(self.instance_classifiers)

            CLAM_outcomes = {
                'Instance_Loss': instance_loss_total,
                'Prediction_Accuracy_Positive': pos_acc_total
            }
        else:
            CLAM_outcomes = {}

        # compute the slide-level representation aggregated per the attention score distribution for the mth class
        SAR = list()
        for i in range(len(img_features)):
            sar = tf.linalg.matmul(tf.transpose(A[i]), h[i])  # return h_[slide,m], shape be (2,512)
            SAR.append(sar)
        slide_agg_rep = tf.add_n(SAR)

        if slide_predict_op:
            CLAM_outcomes.update({
                'Slide_Level_Representation': slide_agg_rep
            })

        # unnormalized slide-level score (s_[slide,m]) with uninitialized entries, shape be (1,num_of_classes)
        slide_score_unnorm = tf.Variable(np.empty((1, self.n_classes)), dtype=tf.float32)

        # return s_[slide,m] (slide-level prediction scores)
        for j in range(self.n_classes):
            ssu = self.bag_classifiers[j](tf.reshape(slide_agg_rep[j], (1, 512)))[0, 0]
            tf.compat.v1.assign(slide_score_unnorm[0, j], ssu)

        Y_hat = tf.math.top_k(slide_score_unnorm, 1)[1][-1]
        Y_prob = tf.compat.v1.math.softmax(slide_score_unnorm)

        return att_net_out, slide_score_unnorm, Y_hat, Y_prob, CLAM_outcomes