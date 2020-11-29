import tensorflow as tf
import tensorflow_addons as tfa
import argparse


def make_arg_parser():
    parser = argparse.ArgumentParser(description='clam command line arguments description',
                                     epilog='epilog')

    parser.add_argument('i', '--is_training',
                        default=True,
                        required=True,
                        help='whether to train the model or not, only executing testing when it is False')

    parser.add_argument('t', '--train_data_dir',
                        dest='train_dir',
                        required=False,
                        help='directory of training tfrecords')

    parser.add_argument('v', '--val_data_dir',
                        dest='val_dir',
                        required=False,
                        help='directory of validation tfrecords')

    parser.add_argument('d', '--test_data_dir',
                        dest='test_dir',
                        required=False,
                        help='directory of testing tfrecords')

    parser.add_argument('r', '--test_result_dir',
                        dest='result_dir',
                        required=False,
                        help='directory of testing result files')

    parser.add_argument('f', '--test_result_file_name',
                        default='clam_test_result_file.tsv',
                        required=False,
                        help='testing result file name')

    parser.add_argument('g', '--train_log_dir',
                        dest='train_log',
                        required=False,
                        help='path to the training log files')

    parser.add_argument('l', '--val_log_dir',
                        dest='val_log',
                        required=False,
                        help='path to the validation log files')

    parser.add_argument('e', '--i_model_dir',
                        dest='i_model_dir',
                        required=False,
                        help='path to where the well-trained instance classifier stored')

    parser.add_argument('s', '--b_model_dir',
                        dest='b_model_dir',
                        required=False,
                        help='path to where the well-trained bag classifier stored')

    parser.add_argument('m', '--c_model_dir',
                        dest='c_model_dir',
                        required=False,
                        help='path to where the well-trained clam model stored')

    parser.add_argument('o', '--i_model_optimizer',
                        default=tfa.optimizers.AdamW,
                        required=False,
                        help='optimizer option for instance classifier')

    parser.add_argument('p', '--b_model_optimizer',
                        default=tfa.optimizers.AdamW,
                        required=False,
                        help='optimizer option for bag classifier')

    parser.add_argument('z', '--c_model_optimizer',
                        default=tfa.optimizers.AdamW,
                        required=False,
                        help='optimizer option for clam model')

    parser.add_argument('y', '--i_loss_func',
                        default=tf.keras.losses.binary_crossentropy,
                        required=False,
                        help='loss function option for instance classifier')

    parser.add_argument('b', '--b_loss_func',
                        default=tf.keras.losses.binary_crossentropy,
                        required=False,
                        help='loss function option for bag classifier')

    parser.add_argument('c', '--c1',
                        default=0.7,
                        required=False,
                        help='scalar of instance loss values')

    parser.add_argument('a', '--c2',
                        default=0.3,
                        required=False,
                        help='scalar of bag loss values')

    parser.add_argument('x', '--att_gate',
                        default=True,
                        required=False,
                        help='whether or not applying gate attention network')

    parser.add_argument('u', '--mut_ex',
                        default=False,
                        required=False,
                        help='whether or not the mutually exclusive assumption holds')

    parser.add_argument('h', '--i_learn_rate',
                        default=2e-04,
                        required=False,
                        help='learning rate for instance classifier')

    parser.add_argument('j', '--b_learn_rate',
                        default=2e-04,
                        required=False,
                        help='learning rate for bag classifier')

    parser.add_argument('k', '--c_learn_rate',
                        default=2e-04,
                        required=False,
                        help='learning rate for clam model')

    parser.add_argument('n', '--i_weight_decay',
                        default=1e-05,
                        required=False,
                        help='L2 weight decay rate for instance classifier')

    parser.add_argument('q', '--b_weight_decay',
                        default=1e-05,
                        required=False,
                        help='L2 weight decay rate for bag classifier')

    parser.add_argument('w', '--c_weight_decay',
                        default=1e-05,
                        required=False,
                        help='L2 weight decay rate for clam model')

    parser.add_argument('S', '--n_class',
                        default=2,
                        required=False,
                        help='number of classes need to be classified, default be 2 for binary classification')

    parser.add_argument('K', '--top_k_percent',
                        default=0.4,
                        required=False,
                        help='percentage of the number of instances from one slide to determine the value of top k')

    parser.add_argument('M', '--m_clam_op',
                        default=False,
                        required=False,
                        help='whether or not applying multi-clam models with multi-bag classifiers included')

    parser.add_argument('B', '--batch_op',
                        default=False,
                        required=False,
                        help='whether or not set batch size during model optimization process')

    parser.add_argument('Z', '--batch_size',
                        default=2000,
                        required=False,
                        help='number of batch size applied during model optimization process')

    parser.add_argument('E', '--epochs',
                        default=200,
                        required=False,
                        help='number of epochs for model optimization process')

    parser.add_argument('W', '--no_warn_op',
                        default=True,
                        required=False,
                        help='whether or not preventing tensorflow from returning warning messages')

    parser.add_argument('O', '--att_only',
                        default=False,
                        required=False,
                        help='if only returned attention score from well-trained model for visualization purposes')

    parser.add_argument('N', '--mil_ins',
                        default=True,
                        required=False,
                        help='whether or not performing instance level clustering')

    parser.add_argument('S', '--net_size',
                        default='big',
                        required=False,
                        help='attention network size which will determine the number of hidden units')

    parser.add_argument('D', '--dropout',
                        default=True,
                        required=False,
                        help='whether or not enabling dropout layer in the attention network')

    parser.add_argument('R', '--dropout_rate',
                        default=0.25,
                        required=False,
                        help='dropout rate for the attention network dropout layer if it is enabled')

    parser.add_argument('F', '--dim_features',
                        default=1024,
                        required=False,
                        help='dimensionality of image feature vectors, default be 1024')

    parser.add_argument('A', '--dim_compress_features',
                        default=512,
                        required=False,
                        help='dimensionality of compressed image feature vectors, default be 512')

    parser.add_argument('H', '--n_hidden_units',
                        default=256,
                        required=False,
                        help='number of hidden unites of each layers in the attention network')

    args = parser.parse_args()

    return args