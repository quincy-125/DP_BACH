import tensorflow as tf

from MODEL.model_attention import G_Att_Net, NG_Att_Net
from MODEL.model_bag_classifier import S_Bag, M_Bag
from MODEL.model_ins_classifier import Ins


class S_CLAM(tf.keras.Model):
    def __init__(self, att_gate=False, net_size='small', top_k_percent=0.2, n_class=2, mut_ex=False,
                 dropout=False, drop_rate=.25, mil_ins=False, att_only=False):
        super(S_CLAM, self).__init__()
        self.att_gate = att_gate
        self.net_size = net_size
        self.top_k_percent = top_k_percent
        self.n_class = n_class
        self.mut_ex = mut_ex
        self.dropout = dropout
        self.drop_rate = drop_rate
        self.mil_ins = mil_ins
        self.att_only = att_only

        self.net_shape_dict = {
            "small": [1024, 512, 256],
            "big": [1024, 512, 384]
        }
        self.net_shape = self.net_shape_dict[self.net_size]

        if self.att_gate:
            self.att_net = G_Att_Net(dim_features=self.net_shape[0],
                                     dim_compress_features=self.net_shape[1],
                                     n_hidden_units=self.net_shape[2],
                                     n_class=self.n_class,
                                     dropout=self.dropout, dropout_rate=self.drop_rate)
        else:
            self.att_net = NG_Att_Net(dim_features=self.net_shape[0],
                                      dim_compress_features=self.net_shape[1],
                                      n_hidden_units=self.net_shape[2],
                                      n_class=self.n_class, dropout=self.dropout,
                                      dropout_rate=self.drop_rate)

        self.ins_net = Ins(dim_compress_features=self.net_shape[1],
                           n_class=self.n_class,
                           top_k_percent=self.top_k_percent,
                           mut_ex=self.mut_ex)

        self.bag_net = S_Bag(dim_compress_features=self.net_shape[1], n_class=self.n_class)

    def networks(self):
        a_net = self.att_net
        i_net = self.ins_net
        b_net = self.bag_net

        c_nets = [a_net, i_net, b_net]

        return c_nets

    def clam_model(self):
        att_model = self.att_net.att_model()
        ins_classifier = self.ins_net.ins_classifier()
        bag_classifier = self.bag_net.bag_classifier()

        clam_model = [att_model, ins_classifier, bag_classifier]

        return clam_model

    def call(self, img_features, slide_label):
        """
        Args:
            img_features -> original 1024-dimensional instance-level feature vectors
            slide_label -> ground-truth slide label, could be 0 or 1 for binary classification
        """

        h, A = self.att_net.call(img_features)
        att_score = A  # output from attention network
        A = tf.math.softmax(A)  # softmax on attention scores

        if self.att_only:
            return att_score

        if self.mil_ins:
            ins_labels, ins_logits_unnorm, ins_logits = self.ins_net.call(slide_label, h, A)

        slide_score_unnorm, Y_hat, Y_prob, predict_slide_label, Y_true = self.bag_net.call(slide_label, A, h)

        return att_score, A, h, ins_labels, ins_logits_unnorm, ins_logits, slide_score_unnorm, \
               Y_prob, Y_hat, Y_true, predict_slide_label


class M_CLAM(tf.keras.Model):
    def __init__(self, att_gate=False, net_size='small', top_k_percent=0.2, n_class=2, mut_ex=False,
                 dropout=False, drop_rate=.25, mil_ins=False, att_only=False):
        super(M_CLAM, self).__init__()
        self.att_gate = att_gate
        self.net_size = net_size
        self.top_k_percent = top_k_percent
        self.n_class = n_class
        self.mut_ex = mut_ex
        self.dropout = dropout
        self.drop_rate = drop_rate
        self.mil_ins = mil_ins
        self.att_only = att_only

        self.net_shape_dict = {
            "small": [1024, 512, 256],
            "big": [1024, 512, 384]
        }
        self.net_shape = self.net_shape_dict[self.net_size]

        if self.att_gate:
            self.att_net = G_Att_Net(dim_features=self.net_shape[0],
                                     dim_compress_features=self.net_shape[1],
                                     n_hidden_units=self.net_shape[2],
                                     n_class=self.n_class,
                                     dropout=self.dropout,
                                     dropout_rate=self.drop_rate)
        else:
            self.att_net = NG_Att_Net(dim_features=self.net_shape[0],
                                      dim_compress_features=self.net_shape[1],
                                      n_hidden_units=self.net_shape[2],
                                      n_class=self.n_class,
                                      dropout=self.dropout,
                                      dropout_rate=self.drop_rate)

        self.ins_net = Ins(dim_compress_features=self.net_shape[1],
                           n_class=self.n_class,
                           top_k_percent=self.top_k_percent,
                           mut_ex=self.mut_ex)

        self.bag_net = M_Bag(dim_compress_features=self.net_shape[1], n_class=self.n_class)

    def networks(self):
        a_net = self.att_net
        i_net = self.ins_net
        b_net = self.bag_net

        c_nets = [a_net, i_net, b_net]

        return c_nets

    def clam_model(self):
        att_model = self.att_net.att_model()
        ins_classifier = self.ins_net.ins_classifier()
        bag_classifier = self.bag_net.bag_classifier()

        clam_model = [att_model, ins_classifier, bag_classifier]

        return clam_model

    def call(self, img_features, slide_label):
        """
        Args:
            img_features -> original 1024-dimensional instance-level feature vectors
            slide_label -> ground-truth slide label, could be 0 or 1 for binary classification
        """

        h, A = self.att_net.call(img_features)
        att_score = A  # output from attention network
        A = tf.math.softmax(A)  # softmax on attention scores

        if self.att_only:
            return att_score

        if self.mil_ins:
            ins_labels, ins_logits_unnorm, ins_logits = self.ins_net.call(slide_label, h, A)

        slide_score_unnorm, Y_hat, Y_prob, predict_slide_label, Y_true = self.bag_net.call(slide_label, A, h)

        return att_score, A, h, ins_labels, ins_logits_unnorm, ins_logits, slide_score_unnorm, \
               Y_prob, Y_hat, Y_true, predict_slide_label