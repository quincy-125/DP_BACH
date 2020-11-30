import argparse

from UTILITY.model_main import clam_main


def make_arg_parser():
    parser = argparse.ArgumentParser(description='clam command line arguments description',
                                     epilog='epilog')

    parser.add_argument('-i', '--is_training_name',
                        type=str,
                        default='True',
                        required=True,
                        help='whether to train the model or not, only executing testing when it is False')

    parser.add_argument('-G', '--multi_gpu_name',
                        type=str,
                        default='False',
                        help='whether or not enabling multiple GPU for model optimization process')

    parser.add_argument('-t', '--train_data_dir',
                        required=False,
                        help='directory of training tfrecords')

    parser.add_argument('-v', '--val_data_dir',
                        required=False,
                        help='directory of validation tfrecords')

    parser.add_argument('-d', '--test_data_dir',
                        required=False,
                        help='directory of testing tfrecords')

    parser.add_argument('-r', '--test_result_dir',
                        required=False,
                        help='directory of testing result files')

    parser.add_argument('-f', '--test_result_file_name',
                        type=str,
                        default='clam_test_result_file.tsv',
                        required=False,
                        help='testing result file name')

    parser.add_argument('-g', '--train_log_dir',
                        required=False,
                        help='path to the training log files')

    parser.add_argument('-l', '--val_log_dir',
                        required=False,
                        help='path to the validation log files')

    parser.add_argument('-e', '--i_model_dir',
                        required=True,
                        help='path to where the well-trained instance classifier stored')

    parser.add_argument('-s', '--b_model_dir',
                        required=True,
                        help='path to where the well-trained bag classifier stored')

    parser.add_argument('-m', '--c_model_dir',
                        required=True,
                        help='path to where the well-trained clam model stored')

    parser.add_argument('-o', '--i_optimizer_name',
                        type=str,
                        default="AdamW",
                        required=False,
                        help='optimizer option for instance classifier')

    parser.add_argument('-p', '--b_optimizer_name',
                        type=str,
                        default="AdamW",
                        required=False,
                        help='optimizer option for bag classifier')

    parser.add_argument('-z', '--c_optimizer_name',
                        type=str,
                        default="AdamW",
                        required=False,
                        help='optimizer option for clam model')

    parser.add_argument('-y', '--i_loss_name',
                        type=str,
                        default="binary_cross_entropy",
                        required=False,
                        help='loss function option for instance classifier')

    parser.add_argument('-b', '--b_loss_name',
                        type=str,
                        default="binary_cross_entropy",
                        required=False,
                        help='loss function option for bag classifier')

    parser.add_argument('-c', '--c_1',
                        type=float,
                        default=0.7,
                        required=False,
                        help='scalar of instance loss values')

    parser.add_argument('-a', '--c_2',
                        type=float,
                        default=0.3,
                        required=False,
                        help='scalar of bag loss values')

    parser.add_argument('-x', '--att_gate_name',
                        type=str,
                        default='True',
                        required=True,
                        help='whether or not applying gate attention network')

    parser.add_argument('-u', '--mut_ex_name',
                        type=str,
                        default='False',
                        required=True,
                        help='whether or not the mutually exclusive assumption holds')

    parser.add_argument('-L', '--i_learn_rate',
                        type=float,
                        default=2e-04,
                        required=False,
                        help='learning rate for instance classifier')

    parser.add_argument('-j', '--b_learn_rate',
                        type=float,
                        default=2e-04,
                        required=False,
                        help='learning rate for bag classifier')

    parser.add_argument('-k', '--c_learn_rate',
                        type=float,
                        default=2e-04,
                        required=False,
                        help='learning rate for clam model')

    parser.add_argument('-n', '--i_weight_decay',
                        type=float,
                        default=1e-05,
                        required=False,
                        help='L2 weight decay rate for instance classifier')

    parser.add_argument('-q', '--b_weight_decay',
                        type=float,
                        default=1e-05,
                        required=False,
                        help='L2 weight decay rate for bag classifier')

    parser.add_argument('-w', '--c_weight_decay',
                        type=float,
                        default=1e-05,
                        required=False,
                        help='L2 weight decay rate for clam model')

    parser.add_argument('-S', '--n_class',
                        type=int,
                        default=2,
                        required=True,
                        help='number of classes need to be classified, default be 2 for binary classification')

    parser.add_argument('-K', '--top_k_percent',
                        type=float,
                        default=0.2,
                        required=True,
                        help='percentage of the number of instances from one slide to determine the value of top k')

    parser.add_argument('-M', '--m_clam_op_name',
                        type=str,
                        default='False',
                        required=True,
                        help='whether or not applying multi-clam models with multi-bag classifiers included')

    parser.add_argument('-B', '--batch_op_name',
                        type=str,
                        default='False',
                        required=False,
                        help='whether or not set batch size during model optimization process')

    parser.add_argument('-Z', '--batch_size',
                        type=int,
                        default=2000,
                        required=False,
                        help='number of batch size applied during model optimization process')

    parser.add_argument('-E', '--epochs',
                        type=int,
                        default=200,
                        required=False,
                        help='number of epochs for model optimization process')

    parser.add_argument('-W', '--no_warn_op_name',
                        type=str,
                        default='True',
                        required=True,
                        help='whether or not preventing tensorflow from returning warning messages')

    parser.add_argument('-O', '--att_only_name',
                        type=str,
                        default='False',
                        required=True,
                        help='if only returned attention score from well-trained model for visualization purposes')

    parser.add_argument('-N', '--mil_ins_name',
                        type=str,
                        default='True',
                        required=True,
                        help='whether or not performing instance level clustering')

    parser.add_argument('-T', '--net_size',
                        type=str,
                        default="big",
                        required=False,
                        help='attention network size which will determine the number of hidden units')

    parser.add_argument('-D', '--dropout_name',
                        type=str,
                        default=True,
                        required=False,
                        help='whether or not enabling dropout layer in the attention network')

    parser.add_argument('-R', '--dropout_rate',
                        type=float,
                        default=0.25,
                        required=False,
                        help='dropout rate for the attention network dropout layer if it is enabled')

    parser.add_argument('-F', '--dim_features',
                        type=int,
                        default=1024,
                        required=False,
                        help='dimensionality of image feature vectors, default be 1024')

    parser.add_argument('-A', '--dim_compress_features',
                        type=int,
                        default=512,
                        required=False,
                        help='dimensionality of compressed image feature vectors, default be 512')

    parser.add_argument('-H', '--n_hidden_units',
                        type=int,
                        default=256,
                        required=False,
                        help='number of hidden unites of each layers in the attention network')

    return parser

def main():
    parser = make_arg_parser()
    args = parser.parse_args()

    clam_main(train_log=args.train_log_dir,
              val_log=args.val_log_dir,
              train_path=args.train_data_dir,
              val_path=args.val_data_dir,
              test_path=args.test_data_dir,
              result_path=args.test_result_dir,
              result_file_name=args.test_result_file_name,
              dim_features=args.dim_features,
              dim_compress_features=args.dim_compress_features,
              n_hidden_units=args.n_hidden_units,
              net_size=args.net_size,
              dropout_name=args.dropout_name,
              dropout_rate=args.dropout_rate,
              i_optimizer_name=args.i_optimizer_name,
              b_optimizer_name=args.b_optimizer_name,
              c_optimizer_name=args.c_optimizer_name,
              i_loss_name=args.i_loss_name,
              b_loss_name=args.b_loss_name,
              mut_ex_name=args.mut_ex_name,
              n_class=args.n_class,
              c1=args.c_1,
              c2=args.c_2,
              i_learn_rate=args.i_learn_rate,
              b_learn_rate=args.b_learn_rate,
              c_learn_rate=args.c_learn_rate,
              i_l2_decay=args.i_weight_decay,
              b_l2_decay=args.b_weight_decay,
              c_l2_decay=args.c_weight_decay,
              top_k_percent=args.top_k_percent,
              batch_size=args.batch_size,
              batch_op_name=args.batch_op_name,
              i_model_dir=args.i_model_dir,
              b_model_dir=args.b_model_dir,
              c_model_dir=args.c_model_dir,
              att_only_name=args.att_only_name,
              mil_ins_name=args.mil_ins_name,
              att_gate_name=args.att_gate_name,
              epochs=args.epochs,
              no_warn_op_name=args.no_warn_op_name,
              m_clam_op_name=args.m_clam_op_name,
              m_gpu_name=args.multi_gpu_name,
              is_training_name=args.is_training_name)
