import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import shutil
import random
import os


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


def most_frequent(List):
    mf = max(set(List), key=List.count)
    return mf


def tf_shut_up(no_warn_op=False):
    if no_warn_op:
        tf.get_logger().setLevel('ERROR')
    else:
        print('Are you sure you want to receive the annoying TensorFlow Warning Messages?', \
              '\n', 'If not, check the value of your input prameter for this function and re-run it.')

def tf_func_options():
    tf_func_dic = {"AdamW": tfa.optimizers.AdamW,
                   "Adam": tf.keras.optimizers.Adam,
                   "binary_cross_entropy": tf.keras.losses.binary_crossentropy,
                   "hinge": tf.keras.losses.hinge}

    return tf_func_dic

def str_to_bool():
    str_bool_dic = {'True': True,
                    'False': False}

    return str_bool_dic

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
    if not os.path.exists(train):
        os.mkdir(os.path.join(path, 'train'))

    if not os.path.exists(valid):
        os.mkdir(os.path.join(path, 'valid'))

    if not os.path.exists(test):
        os.mkdir(os.path.join(path, 'test'))

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

def ng_att_call(ng_att_net, img_features):
    h = list()
    A = list()

    for i in img_features:
        c_imf = ng_att_net[0](i)
        h.append(c_imf)

    for i in img_features:
        a = ng_att_net[1](i)
        A.append(a)
    return h, A


def g_att_call(g_att_net, img_features):
    h = list()
    A = list()

    for i in img_features:
        c_imf = g_att_net[0](i)
        h.append(c_imf)

    for i in img_features:
        layer1_output = g_att_net[1](i)
        layer2_output = g_att_net[2](i)
        a = tf.math.multiply(layer1_output, layer2_output)
        a = g_att_net[3](a)
        A.append(a)

    return h, A


def generate_pos_labels(n_pos_sample):
    return tf.fill(dims=[n_pos_sample, ], value=1)


def generate_neg_labels(n_neg_sample):
    return tf.fill(dims=[n_neg_sample, ], value=0)


def ins_in_call(ins_classifier, h, A_I, top_k_percent, n_class):
    n_ins = top_k_percent * len(h)
    n_ins = int(n_ins)

    pos_label = generate_pos_labels(n_pos_sample=n_ins)
    neg_label = generate_neg_labels(n_neg_sample=n_ins)
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

    for i in range(n_class * n_ins):
        ins_score_unnorm_in = ins_classifier(ins_in[i])
        logit_in = tf.math.softmax(ins_score_unnorm_in)
        logits_unnorm_in.append(ins_score_unnorm_in)
        logits_in.append(logit_in)

    return ins_label_in, logits_unnorm_in, logits_in


def ins_out_call(ins_classifier, h, A_O, top_k_percent):
    n_ins = top_k_percent * len(h)
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
    pos_ins_labels_out = generate_neg_labels(n_neg_sample=n_ins)
    ins_label_out = pos_ins_labels_out

    logits_unnorm_out = list()
    logits_out = list()

    for i in range(n_ins):
        ins_score_unnorm_out = ins_classifier(top_pos[i])
        logit_out = tf.math.softmax(ins_score_unnorm_out)
        logits_unnorm_out.append(ins_score_unnorm_out)
        logits_out.append(logit_out)

    return ins_label_out, logits_unnorm_out, logits_out


def ins_call(m_ins_classifier, bag_label, h, A, n_class, top_k_percent, mut_ex):
    for i in range(n_class):
        ins_classifier = m_ins_classifier[i]
        if i == bag_label:
            A_I = list()
            for j in range(len(A)):
                a_i = A[j][0][i]
                A_I.append(a_i)
            ins_label_in, logits_unnorm_in, logits_in = ins_in_call(ins_classifier=ins_classifier,
                                                                    h=h, A_I=A_I,
                                                                    top_k_percent=top_k_percent,
                                                                    n_class=n_class)
        else:
            if mut_ex:
                A_O = list()
                for j in range(len(A)):
                    a_o = A[j][0][i]
                    A_O.append(a_o)
                ins_label_out, logits_unnorm_out, logits_out = ins_out_call(ins_classifier=ins_classifier,
                                                                            h=h, A_O=A_O,
                                                                            top_k_percent=top_k_percent)
            else:
                continue

    if mut_ex:
        ins_labels = tf.concat(values=[ins_label_in, ins_label_out], axis=0)
        ins_logits_unnorm = logits_unnorm_in + logits_unnorm_out
        ins_logits = logits_in + logits_out
    else:
        ins_labels = ins_label_in
        ins_logits_unnorm = logits_unnorm_in
        ins_logits = logits_in

    return ins_labels, ins_logits_unnorm, ins_logits

def bag_h_slide(A, h):
    # compute the slide-level representation aggregated per the attention score distribution for the mth class
    SAR = list()
    for i in range(len(A)):
        sar = tf.linalg.matmul(tf.transpose(A[i]), h[i])  # shape be (2,512)
        SAR.append(sar)

    slide_agg_rep = tf.math.add_n(SAR)  # return h_[slide,m], shape be (2,512)

    return slide_agg_rep


def s_bag_call(bag_classifier, bag_label, A, h, n_class):
    slide_agg_rep = bag_h_slide(A=A, h=h)

    slide_score_unnorm = bag_classifier(slide_agg_rep)
    slide_score_unnorm = tf.reshape(slide_score_unnorm, (1, n_class))

    Y_hat = tf.math.top_k(slide_score_unnorm, 1)[1][-1]

    Y_prob = tf.math.softmax(tf.reshape(slide_score_unnorm,
                             (1, n_class)))  # shape be (1,2), predictions for each of the classes

    predict_slide_label = np.argmax(Y_prob.numpy())

    Y_true = tf.one_hot([bag_label], 2)

    return slide_score_unnorm, Y_hat, Y_prob, predict_slide_label, Y_true

def m_bag_in_call(bag_classifier, h_slide_I):
    ssu_in = bag_classifier(h_slide_I)[0][0]

    return ssu_in

def m_bag_out_call(bag_classifier, h_slide_O):
    ssu_out = bag_classifier(h_slide_O)[0][0]

    return ssu_out

def m_bag_call(m_bag_classifier, bag_label, A, h, n_class, dim_compress_features):
    slide_agg_rep = bag_h_slide(A=A, h=h)
    # unnormalized slide-level score (s_[slide,m]) with uninitialized entries, shape be (1,num_of_classes)
    slide_score_unnorm = tf.Variable(np.empty((1, n_class)), dtype=tf.float32)
    slide_score_unnorm = tf.reshape(slide_score_unnorm, (1, n_class)).numpy()

    # return s_[slide,m] (slide-level prediction scores)
    for i in range(n_class):
        bag_classifier = m_bag_classifier[i]
        if i == bag_label:
            h_slide_I = tf.reshape(slide_agg_rep[i], (1, dim_compress_features))
            ssu_in = m_bag_in_call(bag_classifier=bag_classifier, h_slide_I=h_slide_I)
        else:
            h_slide_O = tf.reshape(slide_agg_rep[i], (1, dim_compress_features))
            ssu_out = m_bag_out_call(bag_classifier=bag_classifier, h_slide_O=h_slide_O)

    for i in range(n_class):
        if i == bag_label:
            slide_score_unnorm[0, i] = ssu_in
        else:
            slide_score_unnorm[0, i] = ssu_out
    slide_score_unnorm = tf.convert_to_tensor(slide_score_unnorm)

    Y_hat = tf.math.top_k(slide_score_unnorm, 1)[1][-1]
    Y_prob = tf.math.softmax(slide_score_unnorm)
    predict_slide_label = np.argmax(Y_prob.numpy())

    Y_true = tf.one_hot([bag_label], 2)

    return slide_score_unnorm, Y_hat, Y_prob, predict_slide_label, Y_true

def s_clam_call(att_net, ins_net, bag_net, img_features, slide_label,
                n_class, top_k_percent, att_gate, att_only, mil_ins, mut_ex):
    if att_gate:
        h, A = g_att_call(g_att_net=att_net, img_features=img_features)
    else:
        h, A = ng_att_call(ng_att_net=att_net, img_features=img_features)
    att_score = A  # output from attention network
    A = tf.math.softmax(A)   # softmax on attention scores

    if att_only:
        return att_score

    if mil_ins:
        ins_labels, ins_logits_unnorm, ins_logits = ins_call(m_ins_classifier=ins_net,
                                                             bag_label=slide_label,
                                                             h=h, A=A,
                                                             n_class=n_class,
                                                             top_k_percent=top_k_percent,
                                                             mut_ex=mut_ex)

    slide_score_unnorm, Y_hat, Y_prob, predict_slide_label, Y_true = s_bag_call(bag_classifier=bag_net,
                                                                                bag_label=slide_label,
                                                                                A=A, h=h, n_class=n_class)

    return att_score, A, h, ins_labels, ins_logits_unnorm, ins_logits, \
           slide_score_unnorm, Y_prob, Y_hat, Y_true, predict_slide_label

def m_clam_call(att_net, ins_net, bag_net, img_features, slide_label,
                n_class, dim_compress_features, top_k_percent, att_gate, att_only, mil_ins, mut_ex):
    if att_gate:
        h, A = g_att_call(g_att_net=att_net, img_features=img_features)
    else:
        h, A = ng_att_call(ng_att_net=att_net, img_features=img_features)
    att_score = A  # output from attention network
    A = tf.math.softmax(A)  # softmax on attention scores

    if att_only:
        return att_score

    if mil_ins:
        ins_labels, ins_logits_unnorm, ins_logits = ins_call(m_ins_classifier=ins_net,
                                                             bag_label=slide_label,
                                                             h=h, A=A,
                                                             n_class=n_class,
                                                             top_k_percent=top_k_percent,
                                                             mut_ex=mut_ex)

    slide_score_unnorm, Y_hat, Y_prob, \
    predict_slide_label, Y_true = m_bag_call(m_bag_classifier=bag_net, bag_label=slide_label,
                                             A=A, h=h, n_class=n_class,
                                             dim_compress_features=dim_compress_features)

    return att_score, A, h, ins_labels, ins_logits_unnorm, ins_logits, \
           slide_score_unnorm, Y_prob, Y_hat, Y_true, predict_slide_label

def multi_gpu_train(model):
    # In TF V2.0, eager_execution will prevent from enabling multi-gpu for training
    tf.compat.v1.disable_eager_execution()

    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        parallel_model = model

    return parallel_model

def model_save(i_model, b_model, c_model, i_model_dir, b_model_dir,
               c_model_dir, n_class, m_clam_op, att_gate):

    for i in range(n_class):
        i_model.ins_classifier()[i].save(os.path.join(i_model_dir, 'M_Ins', 'Class_' + str(i)))

    if m_clam_op:
        for j in range(n_class):
            b_model.bag_classifier()[j].save(os.path.join(b_model_dir, 'M_Bag', 'Class_' + str(j)))
    else:
        b_model.bag_classifier().save(os.path.join(b_model_dir, 'S_Bag'))

    clam_model_names = ['_Att', '_Ins', '_Bag']

    if m_clam_op:
        if att_gate:
            att_nets = c_model.clam_model()[0]
            for m in range(len(att_nets)):
                att_nets[m].save(os.path.join(c_model_dir, 'G' + clam_model_names[0], 'Model_' + str(m + 1)))
        else:
            att_nets = c_model.clam_model()[0]
            for m in range(len(att_nets)):
                att_nets[m].save(os.path.join(c_model_dir, 'NG' + clam_model_names[0], 'Model_' + str(m + 1)))

        for n in range(n_class):
            ins_nets = c_model.clam_model()[1]
            bag_nets = c_model.clam_model()[2]

            ins_nets[n].save(os.path.join(c_model_dir, 'M' + clam_model_names[1], 'Class_' + str(n)))
            bag_nets[n].save(os.path.join(c_model_dir, 'M' + clam_model_names[2], 'Class_' + str(n)))
    else:
        if att_gate:
            att_nets = c_model.clam_model()[0]
            for m in range(len(att_nets)):
                att_nets[m].save(os.path.join(c_model_dir, 'G' + clam_model_names[0], 'Model_' + str(m + 1)))
        else:
            att_nets = c_model.clam_model()[0]
            for m in range(len(att_nets)):
                att_nets[m].save(os.path.join(c_model_dir, 'NG' + clam_model_names[0], 'Model_' + str(m + 1)))

        for n in range(n_class):
            ins_nets = c_model.clam_model()[1]
            ins_nets[n].save(os.path.join(c_model_dir, 'M' + clam_model_names[1], 'Class_' + str(n)))

        c_model.clam_model()[2].save(os.path.join(c_model_dir, 'S' + clam_model_names[2]))


def restore_model(i_model_dir, b_model_dir, c_model_dir, n_class,
                  m_clam_op, att_gate):

    i_trained_model = list()
    for i in range(n_class):
        m_ins_names = os.listdir(os.path.join(i_model_dir, 'M_Ins'))
        m_ins_names.sort()
        m_ins_name = m_ins_names[i]
        m_ins_model = tf.keras.models.load_model(os.path.join(i_model_dir, 'M_Ins', m_ins_name))
        i_trained_model.append(m_ins_model)

    if m_clam_op:
        b_trained_model = list()
        for j in range(n_class):
            m_bag_names = os.listdir(os.path.join(b_model_dir, 'M_Bag'))
            m_bag_names.sort()
            m_bag_name = m_bag_names[j]
            m_bag_model = tf.keras.models.load_model(os.path.join(b_model_dir, 'M_Bag', m_bag_name))
            b_trained_model.append(m_bag_model)
    else:
        s_bag_name = os.listdir(b_model_dir)[0]
        b_trained_model = tf.keras.models.load_model(os.path.join(b_model_dir, s_bag_name))

    clam_model_names = ['_Att', '_Ins', '_Bag']

    trained_att_net = list()
    trained_ins_classifier = list()
    trained_bag_classifier = list()

    c_trained_model = list()

    if m_clam_op:
        if att_gate:
            att_nets_dir = os.path.join(c_model_dir, 'G' + clam_model_names[0])
            for k in range(len(os.listdir(att_nets_dir))):
                att_net = tf.keras.models.load_model(os.path.join(att_nets_dir, 'Model_' + str(k + 1)))
                trained_att_net.append(att_net)
        else:
            att_nets_dir = os.path.join(c_model_dir, 'NG' + clam_model_names[0])
            for k in range(len(os.listdir(att_nets_dir))):
                att_net = tf.keras.models.load_model(os.path.join(att_nets_dir, 'Model_' + str(k + 1)))
                trained_att_net.append(att_net)

        ins_nets_dir = os.path.join(c_model_dir, 'M' + clam_model_names[1])
        bag_nets_dir = os.path.join(c_model_dir, 'M' + clam_model_names[2])

        for m in range(n_class):
            ins_net = tf.keras.models.load_model(os.path.join(ins_nets_dir, 'Class_' + str(m)))
            bag_net = tf.keras.models.load_model(os.path.join(bag_nets_dir, 'Class_' + str(m)))

            trained_ins_classifier.append(ins_net)
            trained_bag_classifier.append(bag_net)

        c_trained_model = [trained_att_net, trained_ins_classifier, trained_bag_classifier]
    else:
        if att_gate:
            att_nets_dir = os.path.join(c_model_dir, 'G' + clam_model_names[0])
            for k in range(len(os.listdir(att_nets_dir))):
                att_net = tf.keras.models.load_model(os.path.join(att_nets_dir, 'Model_' + str(k + 1)))
                trained_att_net.append(att_net)
        else:
            att_nets_dir = os.path.join(c_model_dir, 'NG' + clam_model_names[0])
            for k in range(len(os.listdir(att_nets_dir))):
                att_net = tf.keras.models.load_model(os.path.join(att_nets_dir, 'Model_' + str(k + 1)))
                trained_att_net.append(att_net)

        ins_nets_dir = os.path.join(c_model_dir, 'M' + clam_model_names[1])

        for m in range(n_class):
            ins_net = tf.keras.models.load_model(os.path.join(ins_nets_dir, 'Class_' + str(m)))
            trained_ins_classifier.append(ins_net)

        bag_nets_dir = os.path.join(c_model_dir, 'S' + clam_model_names[2])
        trained_bag_classifier.append(tf.keras.models.load_model(bag_nets_dir))

        c_trained_model = [trained_att_net, trained_ins_classifier, trained_bag_classifier[0]]

    return i_trained_model, b_trained_model, c_trained_model