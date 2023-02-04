import yaml
import json

from yaml.loader import SafeLoader
from UTILITY.model_main import clam_main


def load_args_config():
    with open("configs/train_config.yaml")

def main(args):
    clam_main(
        train_log=args.train_log_dir,
        val_log=args.val_log_dir,
        train_path=args.train_data_dir,
        val_path=args.val_data_dir,
        test_path=args.test_data_dir,
        result_path=args.test_result_dir,
        result_file_name=args.test_result_file_name,
        imf_norm_op_name=args.imf_norm_op_name,
        dim_compress_features=args.dim_compress_features,
        net_size=args.net_size,
        dropout_name=args.dropout_name,
        dropout_rate=args.dropout_rate,
        i_optimizer_name=args.i_optimizer_name,
        b_optimizer_name=args.b_optimizer_name,
        a_optimizer_name=args.a_optimizer_name,
        i_loss_name=args.i_loss_name,
        b_loss_name=args.b_loss_name,
        mut_ex_name=args.mut_ex_name,
        n_class=args.n_class,
        c1=args.c_1,
        c2=args.c_2,
        i_learn_rate=args.i_learn_rate,
        b_learn_rate=args.b_learn_rate,
        a_learn_rate=args.a_learn_rate,
        i_l2_decay=args.i_weight_decay,
        b_l2_decay=args.b_weight_decay,
        a_l2_decay=args.a_weight_decay,
        top_k_percent=args.top_k_percent,
        batch_size=args.batch_size,
        batch_op_name=args.batch_op_name,
        c_model_dir=args.c_model_dir,
        att_only_name=args.att_only_name,
        mil_ins_name=args.mil_ins_name,
        att_gate_name=args.att_gate_name,
        epochs=args.epochs,
        n_test_steps=args.test_steps,
        no_warn_op_name=args.no_warn_op_name,
        i_wd_op_name=args.i_wd_op_name,
        b_wd_op_name=args.b_wd_op_name,
        a_wd_op_name=args.a_wd_op_name,
        m_clam_op_name=args.m_clam_op_name,
        is_training_name=args.is_training_name,
        m_gpu_op_name=args.multi_gpu_name,
    )
