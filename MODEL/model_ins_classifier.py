import tensorflow as tf


class Ins(tf.keras.Model):
    def __init__(self, dim_compress_features=512, n_class=2, top_k_percent=0.2, mut_ex=False):
        super(Ins, self).__init__()
        self.dim_compress_features = dim_compress_features
        self.n_class = n_class
        self.top_k_percent = top_k_percent
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
        n_ins = self.top_k_percent * len(h)
        n_ins = int(n_ins)

        pos_label = self.generate_pos_labels(n_ins)
        neg_label = self.generate_neg_labels(n_ins)
        ins_label_in = tf.concat(values=[pos_label, neg_label], axis=0)
        A_I = tf.reshape(tf.convert_to_tensor(A_I), (1, len(A_I)))

        top_pos_ids = tf.math.top_k(A_I, n_ins)[1][-1]
        pos_index = list()
        for i in top_pos_ids:
            pos_index.append(i)

        pos_index = tf.convert_to_tensor(pos_index)
        top_pos = list()
        for i in pos_index:
            top_pos.append(h[i])

        top_neg_ids = tf.math.top_k(-A_I, n_ins)[1][-1]
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

        for i in range(self.n_class * n_ins):
            ins_score_unnorm_in = ins_classifier(ins_in[i])
            logit_in = tf.math.softmax(ins_score_unnorm_in)
            logits_unnorm_in.append(ins_score_unnorm_in)
            logits_in.append(logit_in)

        return ins_label_in, logits_unnorm_in, logits_in

    def out_call(self, ins_classifier, h, A_O):
        n_ins = self.top_k_percent * len(h)
        n_ins = int(n_ins)

        # get compressed 512-dimensional instance-level feature vectors for following use, denoted by h
        A_O = tf.reshape(tf.convert_to_tensor(A_O), (1, len(A_O)))
        top_pos_ids = tf.math.top_k(A_O, n_ins)[1][-1]
        pos_index = list()
        for i in top_pos_ids:
            pos_index.append(i)

        pos_index = tf.convert_to_tensor(pos_index)
        top_pos = list()
        for i in pos_index:
            top_pos.append(h[i])

        # mutually-exclusive -> top k instances w/ highest attention scores ==> false pos = neg
        pos_ins_labels_out = self.generate_neg_labels(n_ins)
        ins_label_out = pos_ins_labels_out

        logits_unnorm_out = list()
        logits_out = list()

        for i in range(n_ins):
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