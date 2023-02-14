import tensorflow as tf


def write_tb(
    writer,
    step,
    neg_dist,
    pos_dist,
    total_dist,
    percent_correct,
    siamese_net,
    neg_hist,
    pos_hist,
    training_acc,
    testing_acc,
):
    with writer.as_default():
        tf.summary.scalar("neg_dist", neg_dist, step=step)
        tf.summary.scalar("pos_dist", pos_dist, step=step)
        tf.summary.scalar("total_dist", total_dist, step=step)
        tf.summary.scalar("training_accuracy", training_acc, step=step)
        tf.summary.scalar("validation_accuracy", testing_acc, step=step)
        tf.summary.scalar("percent_correct", percent_correct, step=step)
        tf.summary.histogram("Negative_Embeddings", neg_hist, step=step)
        tf.summary.histogram("Positive_Embeddings", pos_hist, step=step)
        for i in siamese_net.layers[3].layers:
            for j in i.trainable_variables:
                tf.summary.histogram(j.name, j, step=step)
