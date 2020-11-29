import tensorflow as tf

from UTILITY.model_main import clam_main
from cli import make_arg_parser


if __name__ == '__main__':
    print(tf.__version__)
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print(tf.config.experimental.list_physical_devices('GPU'))
    assert len(tf.config.experimental.list_physical_devices('GPU')) > 0, "No GPUs found"

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
              dropout=args.dropout,
              dropout_rate=args.dropout_rate,
              i_optimizer_func=args.i_model_optimizer,
              b_optimizer_func=args.b_model_optimizer,
              c_optimizer_func=args.c_model_optimizer,
              i_loss_func=args.i_loss_func,
              b_loss_func=args.b_loss_func,
              mut_ex=args.mut_ex,
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
              batch_op=args.batch_op,
              i_model_dir=args.i_model_dir,
              b_model_dir=args.b_model_dir,
              c_model_dir=args.c_model_dir,
              att_only=args.att_only,
              mil_ins=args.mil_ins,
              att_gate=args.att_gate,
              epochs=args.epochs,
              no_warn_op=args.no_warn_op,
              m_clam_op=args.m_clam_op,
              is_training=args.is_training)

