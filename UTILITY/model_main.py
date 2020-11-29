import tensorflow as tf
import time

from UTILITY.model_train import train_step
from UTILITY.model_val import val_step
from UTILITY.model_test import test_step
from UTILITY.util import model_save, restore_model


def train_val(train_log, val_log, train_path, val_path, i_model, b_model,
              c_model, i_optimizer_func, b_optimizer_func, c_optimizer_func,
              i_loss_func, b_loss_func, mut_ex, n_class, c1, c2,
              i_learn_rate, b_learn_rate, c_learn_rate,
              i_l2_decay, b_l2_decay, c_l2_decay, n_ins,
              batch_size, batch_op, epochs):
    train_summary_writer = tf.summary.create_file_writer(train_log)
    val_summary_writer = tf.summary.create_file_writer(val_log)

    for epoch in range(epochs):
        # Training Step
        start_time = time.time()

        train_loss, train_ins_loss, train_bag_loss, train_tn, train_fp, train_fn, train_tp, \
        train_sensitivity, train_specificity, train_acc, train_auc = train_step(
            i_model=i_model, b_model=b_model, c_model=c_model, train_path=train_path,
            i_optimizer_func=i_optimizer_func, b_optimizer_func=b_optimizer_func,
            c_optimizer_func=c_optimizer_func, i_loss_func=i_loss_func,
            b_loss_func=b_loss_func, mut_ex=mut_ex, n_class=n_class,
            c1=c1, c2=c2, i_learn_rate=i_learn_rate, b_learn_rate=b_learn_rate,
            c_learn_rate=c_learn_rate, i_l2_decay=i_l2_decay, b_l2_decay=b_l2_decay,
            c_l2_decay=c_l2_decay, n_ins=n_ins, batch_size=batch_size, batch_op=batch_op)

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
        val_sensitivity, val_specificity, val_acc, val_auc = val_step(
            i_model=i_model, b_model=b_model, c_model=c_model, val_path=val_path,
            i_loss_func=i_loss_func, b_loss_func=b_loss_func, mut_ex=mut_ex,
            n_class=n_class, c1=c1, c2=c2, n_ins=n_ins, batch_size=batch_size, batch_op=batch_op)

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
        template = '\n Epoch {},  Train Loss: {}, Train Accuracy: {}, Val Loss: {}, Val Accuracy: {}, Epoch Running ' \
                   'Time: {} '
        print(template.format(epoch + 1,
                              f"{float(train_loss):.8}",
                              f"{float(train_acc):.4%}",
                              f"{float(val_loss):.8}",
                              f"{float(val_acc):.4%}",
                              "--- %s mins ---" % int(epoch_run_time / 60)))


def clam_optimize(train_log, val_log, train_path, val_path, i_model, b_model,
                  c_model, i_optimizer_func, b_optimizer_func, c_optimizer_func,
                  i_loss_func, b_loss_func, mut_ex, n_class, c1, c2,
                  i_learn_rate, b_learn_rate, c_learn_rate, i_l2_decay, b_l2_decay,
                  c_l2_decay, n_ins, batch_size, batch_op, i_model_dir, b_model_dir,
                  c_model_dir, m_bag_op, m_clam_op, att_gate, epochs):

    train_val(train_log=train_log, val_log=val_log, train_path=train_path,
              val_path=val_path, i_model=i_model, b_model=b_model, c_model=c_model,
              i_optimizer_func=i_optimizer_func, b_optimizer_func=b_optimizer_func,
              c_optimizer_func=c_optimizer_func, i_loss_func=i_loss_func,
              b_loss_func=b_loss_func, mut_ex=mut_ex, n_class=n_class,
              c1=c1, c2=c2, i_learn_rate=i_learn_rate, b_learn_rate=b_learn_rate,
              c_learn_rate=c_learn_rate, i_l2_decay=i_l2_decay, b_l2_decay=b_l2_decay,
              c_l2_decay=c_l2_decay, n_ins=n_ins, batch_size=batch_size,
              batch_op=batch_op, epochs=epochs)

    model_save(i_model=i_model, b_model=b_model, c_model=c_model,
               i_model_dir=i_model_dir, b_model_dir=b_model_dir,
               c_model_dir=c_model_dir, n_class=n_class, m_bag_op=m_bag_op,
               m_clam_op=m_clam_op, att_gate=att_gate)

def clam_test(n_class, n_ins, att_gate, att_only, mil_ins, mut_ex, test_path,
              result_path, result_file_name, i_model_dir, b_model_dir, c_model_dir,
              m_bag_op, m_clam_op):

    i_trained_model, b_trained_model, c_trained_model = restore_model(i_model_dir=i_model_dir,
                                                                      b_model_dir=b_model_dir,
                                                                      c_model_dir=c_model_dir,
                                                                      n_class=n_class, m_bag_op=m_bag_op,
                                                                      m_clam_op=m_clam_op, att_gate=att_gate)

    test_step(n_class=n_class, n_ins=n_ins,
              att_gate=att_gate, att_only=att_only,
              mil_ins=mil_ins, mut_ex=mut_ex,
              i_model=i_trained_model,
              b_model=b_trained_model,
              c_model=c_trained_model,
              test_path=test_path,
              result_path=result_path,
              result_file_name=result_file_name)

def clam_main(train_log, val_log, train_path, val_path, test_path,
              result_path, result_file_name,
              i_model, b_model, c_model,
              i_optimizer_func, b_optimizer_func, c_optimizer_func,
              i_loss_func, b_loss_func, mut_ex, n_class, c1, c2,
              i_learn_rate, b_learn_rate, c_learn_rate, i_l2_decay, b_l2_decay,
              c_l2_decay, n_ins, batch_size, batch_op, i_model_dir, b_model_dir,
              att_only, mil_ins, c_model_dir, m_bag_op, m_clam_op, att_gate,
              epochs, is_training=False):

    if is_training:
        clam_optimize(train_log=train_log, val_log=val_log,
                      train_path=train_path, val_path=val_path,
                      i_model=i_model, b_model=b_model, c_model=c_model,
                      i_optimizer_func=i_optimizer_func,
                      b_optimizer_func=b_optimizer_func,
                      c_optimizer_func=c_optimizer_func,
                      i_loss_func=i_loss_func,
                      b_loss_func=b_loss_func,
                      mut_ex=mut_ex,
                      n_class=n_class,
                      c1=c1, c2=c2,
                      i_learn_rate=i_learn_rate,
                      b_learn_rate=b_learn_rate,
                      c_learn_rate=c_learn_rate,
                      i_l2_decay=i_l2_decay,
                      b_l2_decay=b_l2_decay,
                      c_l2_decay=c_l2_decay,
                      n_ins=n_ins,
                      batch_size=batch_size, batch_op=batch_op,
                      i_model_dir=i_model_dir,
                      b_model_dir=b_model_dir,
                      c_model_dir=c_model_dir,
                      m_bag_op=m_bag_op,
                      m_clam_op=m_clam_op,
                      att_gate=att_gate,
                      epochs=epochs)
    else:
        clam_test(n_class=n_class, n_ins=n_ins,
                  att_gate=att_gate, att_only=att_only,
                  mil_ins=mil_ins, mut_ex=mut_ex,
                  test_path=test_path,
                  result_path=result_path,
                  result_file_name=result_file_name,
                  i_model_dir=i_model_dir,
                  b_model_dir=b_model_dir,
                  c_model_dir=c_model_dir,
                  m_bag_op=m_bag_op, m_clam_op=m_clam_op)
