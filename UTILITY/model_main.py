import tensorflow as tf
import time

from MODEL.model_clam import S_CLAM, M_CLAM
from UTILITY.model_train import train_step
from UTILITY.model_val import val_step
from UTILITY.model_test import test_step
from UTILITY.util import model_save, restore_model, tf_shut_up, str_to_bool, multi_gpu_train


def train_val(train_log, val_log, train_path, val_path, imf_norm_op, c_model,
              i_wd_op_name, b_wd_op_name, a_wd_op_name,
              i_optimizer_name, b_optimizer_name, a_optimizer_name,
              i_loss_name, b_loss_name, mut_ex, n_class, c1, c2,
              i_learn_rate, b_learn_rate, a_learn_rate,
              i_l2_decay, b_l2_decay, a_l2_decay, top_k_percent,
              batch_size, batch_op, epochs):

    train_summary_writer = tf.summary.create_file_writer(train_log)
    val_summary_writer = tf.summary.create_file_writer(val_log)

    for epoch in range(epochs):
        # Training Step
        start_time = time.time()

        train_loss, train_ins_loss, train_bag_loss, train_tn, train_fp, train_fn, train_tp, \
        train_sensitivity, train_specificity, \
        train_acc, train_auc = train_step(c_model=c_model,
                                          train_path=train_path,
                                          imf_norm_op=imf_norm_op,
                                          i_wd_op_name=i_wd_op_name,
                                          b_wd_op_name=b_wd_op_name,
                                          a_wd_op_name=a_wd_op_name,
                                          i_optimizer_name=i_optimizer_name,
                                          b_optimizer_name=b_optimizer_name,
                                          a_optimizer_name=a_optimizer_name,
                                          i_loss_name=i_loss_name,
                                          b_loss_name=b_loss_name,
                                          mut_ex=mut_ex,
                                          n_class=n_class,
                                          c1=c1, c2=c2,
                                          i_learn_rate=i_learn_rate,
                                          b_learn_rate=b_learn_rate,
                                          a_learn_rate=a_learn_rate,
                                          i_l2_decay=i_l2_decay,
                                          b_l2_decay=b_l2_decay,
                                          a_l2_decay=a_l2_decay,
                                          top_k_percent=top_k_percent,
                                          batch_size=batch_size,
                                          batch_op=batch_op)

        with train_summary_writer.as_default():
            tf.summary.scalar('Total Loss', float(train_loss), step=epoch)
            tf.summary.scalar('Instance Loss', float(train_ins_loss), step=epoch)
            tf.summary.scalar('Bag Loss', float(train_bag_loss), step=epoch)
            tf.summary.scalar('Accuracy', float(train_acc), step=epoch)
            tf.summary.scalar('AUC', float(train_auc), step=epoch)
            tf.summary.scalar('Sensitivity', float(train_sensitivity), step=epoch)
            tf.summary.scalar('Specificity', float(train_specificity), step=epoch)
            tf.summary.histogram('True Positive', int(train_tp), step=epoch)
            tf.summary.histogram('False Positive', int(train_fp), step=epoch)
            tf.summary.histogram('True Negative', int(train_tn), step=epoch)
            tf.summary.histogram('False Negative', int(train_fn), step=epoch)

        # Validation Step
        val_loss, val_ins_loss, val_bag_loss, val_tn, val_fp, val_fn, val_tp, \
        val_sensitivity, val_specificity, \
        val_acc, val_auc = val_step(c_model=c_model,
                                    val_path=val_path,
                                    imf_norm_op=imf_norm_op,
                                    i_loss_name=i_loss_name,
                                    b_loss_name=b_loss_name,
                                    mut_ex=mut_ex,
                                    n_class=n_class,
                                    c1=c1, c2=c2,
                                    top_k_percent=top_k_percent,
                                    batch_size=batch_size,
                                    batch_op=batch_op)

        with val_summary_writer.as_default():
            tf.summary.scalar('Total Loss', float(val_loss), step=epoch)
            tf.summary.scalar('Instance Loss', float(val_ins_loss), step=epoch)
            tf.summary.scalar('Bag Loss', float(val_bag_loss), step=epoch)
            tf.summary.scalar('Accuracy', float(val_acc), step=epoch)
            tf.summary.scalar('AUC', float(val_auc), step=epoch)
            tf.summary.scalar('Sensitivity', float(val_sensitivity), step=epoch)
            tf.summary.scalar('Specificity', float(val_specificity), step=epoch)
            tf.summary.histogram('True Positive', int(val_tp), step=epoch)
            tf.summary.histogram('False Positive', int(val_fp), step=epoch)
            tf.summary.histogram('True Negative', int(val_tn), step=epoch)
            tf.summary.histogram('False Negative', int(val_fn), step=epoch)

        epoch_run_time = time.time() - start_time

        # early-stopping
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1E-02, patience=20,
                                         mode='min', restore_best_weights=True)

        template = '\n Epoch {},  Train Loss: {}, Train Accuracy: {}, Val Loss: {}, Val Accuracy: {}, Epoch Running ' \
                   'Time: {} '
        print(template.format(epoch + 1,
                              f"{float(train_loss):.8}",
                              f"{float(train_acc):.4%}",
                              f"{float(val_loss):.8}",
                              f"{float(val_acc):.4%}",
                              "--- %s mins ---" % int(epoch_run_time / 60)))


def clam_optimize(train_log, val_log, train_path, val_path, imf_norm_op, c_model,
                  i_wd_op_name, b_wd_op_name, a_wd_op_name,
                  i_optimizer_name, b_optimizer_name, a_optimizer_name,
                  i_loss_name, b_loss_name, mut_ex, n_class, c1, c2,
                  i_learn_rate, b_learn_rate, a_learn_rate, i_l2_decay, b_l2_decay,
                  a_l2_decay, top_k_percent, batch_size, batch_op,
                  c_model_dir, m_clam_op, att_gate, epochs):

    train_val(train_log=train_log, val_log=val_log, train_path=train_path,
              val_path=val_path, imf_norm_op=imf_norm_op, c_model=c_model,
              i_wd_op_name=i_wd_op_name, b_wd_op_name=b_wd_op_name, a_wd_op_name=a_wd_op_name,
              i_optimizer_name=i_optimizer_name, b_optimizer_name=b_optimizer_name,
              a_optimizer_name=a_optimizer_name, i_loss_name=i_loss_name,
              b_loss_name=b_loss_name, mut_ex=mut_ex, n_class=n_class,
              c1=c1, c2=c2, i_learn_rate=i_learn_rate, b_learn_rate=b_learn_rate,
              a_learn_rate=a_learn_rate, i_l2_decay=i_l2_decay, b_l2_decay=b_l2_decay,
              a_l2_decay=a_l2_decay, top_k_percent=top_k_percent,
              batch_size=batch_size, batch_op=batch_op, epochs=epochs)

    model_save(c_model=c_model, c_model_dir=c_model_dir, n_class=n_class,
               m_clam_op=m_clam_op, att_gate=att_gate)

def clam_test(n_class, top_k_percent, att_gate, att_only, mil_ins, mut_ex, test_path,
              result_path, result_file_name, c_model_dir,
              dim_compress_features, imf_norm_op, m_clam_op, n_test_steps):

    c_trained_model = restore_model(c_model_dir=c_model_dir,
                                    n_class=n_class,
                                    m_clam_op=m_clam_op,
                                    att_gate=att_gate)

    test_step(n_class=n_class,
              top_k_percent=top_k_percent,
              att_gate=att_gate,
              att_only=att_only,
              mil_ins=mil_ins,
              mut_ex=mut_ex,
              m_clam_op=m_clam_op,
              imf_norm_op=imf_norm_op,
              c_model=c_trained_model,
              dim_compress_features=dim_compress_features,
              test_path=test_path,
              result_path=result_path,
              result_file_name=result_file_name,
              n_test_steps=n_test_steps)

def load_model(n_class, top_k_percent, net_size, mut_ex, att_gate, att_only,
               mil_ins, dropout, dropout_rate, m_gpu):

    s_clam = S_CLAM(att_gate=att_gate,
                    net_size=net_size,
                    top_k_percent=top_k_percent,
                    n_class=n_class,
                    mut_ex=mut_ex,
                    dropout=dropout,
                    drop_rate=dropout_rate,
                    mil_ins=mil_ins,
                    att_only=att_only)

    m_clam = M_CLAM(att_gate=att_gate,
                    net_size=net_size,
                    top_k_percent=top_k_percent,
                    n_class=n_class,
                    mut_ex=mut_ex,
                    dropout=dropout,
                    drop_rate=dropout_rate,
                    mil_ins=mil_ins,
                    att_only=att_only)
    if m_gpu:
        s_clam_model = multi_gpu_train(s_clam)
        m_clam_model = multi_gpu_train(m_clam)
    else:
        s_clam_model = s_clam
        m_clam_model = m_clam

    c_model = [s_clam_model, m_clam_model]

    return c_model

def clam_main(train_log, val_log, train_path, val_path, test_path,
              result_path, result_file_name, imf_norm_op_name,
              dim_compress_features, net_size, dropout_name, dropout_rate,
              i_optimizer_name, b_optimizer_name, a_optimizer_name,
              i_loss_name, b_loss_name, mut_ex_name, n_class, c1, c2,
              i_learn_rate, b_learn_rate, a_learn_rate,
              i_l2_decay, b_l2_decay, a_l2_decay,
              top_k_percent, batch_size, batch_op_name,
              c_model_dir, att_only_name, mil_ins_name, att_gate_name,
              epochs, n_test_steps,
              no_warn_op_name,
              i_wd_op_name, b_wd_op_name, a_wd_op_name,
              m_clam_op_name='False',
              m_gpu_name='False',
              is_training_name='True'):

    str_bool_dic = str_to_bool()

    imf_norm_op = str_bool_dic[imf_norm_op_name]
    m_gpu = str_bool_dic[m_gpu_name]
    dropout = str_bool_dic[dropout_name]
    mut_ex = str_bool_dic[mut_ex_name]
    batch_op = str_bool_dic[batch_op_name]
    att_only = str_bool_dic[att_only_name]
    mil_ins = str_bool_dic[mil_ins_name]
    att_gate = str_bool_dic[att_gate_name]
    no_warn_op = str_bool_dic[no_warn_op_name]
    m_clam_op = str_bool_dic[m_clam_op_name]
    is_training = str_bool_dic[is_training_name]

    if is_training:
        c_model = load_model(n_class=n_class,
                             top_k_percent=top_k_percent,
                             net_size=net_size,
                             mut_ex=mut_ex,
                             att_gate=att_gate,
                             att_only=att_only,
                             mil_ins=mil_ins,
                             dropout=dropout,
                             dropout_rate=dropout_rate,
                             m_gpu=m_gpu)

        tf_shut_up(no_warn_op=no_warn_op)

        if m_clam_op:
            b_c_model_index = 1
        else:
            b_c_model_index = 0

        clam_optimize(train_log=train_log, val_log=val_log,
                      train_path=train_path, val_path=val_path,
                      imf_norm_op=imf_norm_op,
                      c_model=c_model[b_c_model_index],
                      i_wd_op_name=i_wd_op_name,
                      b_wd_op_name=b_wd_op_name,
                      a_wd_op_name=a_wd_op_name,
                      i_optimizer_name=i_optimizer_name,
                      b_optimizer_name=b_optimizer_name,
                      a_optimizer_name=a_optimizer_name,
                      i_loss_name=i_loss_name,
                      b_loss_name=b_loss_name,
                      mut_ex=mut_ex,
                      n_class=n_class,
                      c1=c1, c2=c2,
                      i_learn_rate=i_learn_rate,
                      b_learn_rate=b_learn_rate,
                      a_learn_rate=a_learn_rate,
                      i_l2_decay=i_l2_decay,
                      b_l2_decay=b_l2_decay,
                      a_l2_decay=a_l2_decay,
                      top_k_percent=top_k_percent,
                      batch_size=batch_size, batch_op=batch_op,
                      c_model_dir=c_model_dir,
                      m_clam_op=m_clam_op,
                      att_gate=att_gate,
                      epochs=epochs)
    else:
        clam_test(n_class=n_class,
                  top_k_percent=top_k_percent,
                  att_gate=att_gate,
                  att_only=att_only,
                  mil_ins=mil_ins,
                  mut_ex=mut_ex,
                  test_path=test_path,
                  result_path=result_path,
                  result_file_name=result_file_name,
                  c_model_dir=c_model_dir,
                  dim_compress_features=dim_compress_features,
                  imf_norm_op=imf_norm_op,
                  m_clam_op=m_clam_op,
                  n_test_steps=n_test_steps)
