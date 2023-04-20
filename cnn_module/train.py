from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
import os
import sys
import glob
import re
import tensorflow as tf

# tf.keras.backend.clear_session()
from callbacks import CallBacks
from cnn_module.training_module.model_factory import GetModel
from cnn_module.training_module.util import (
    Preprocess,
    format_example,
    format_example_tf,
    update_status,
)


def train_main(args):
    logging.basicConfig(
        stream=sys.stderr,
        level=args.verbose,
        format="%(name)s (%(levelname)s): %(message)s",
    )

    logger = logging.getLogger(__name__)
    logger.setLevel(args.logLevel)

    # Get Training and Validation data
    train_data = Preprocess(
        args.image_dir_train,
        args.filetype,
        args.tfrecord_image,
        args.tfrecord_label,
        loss_function=args.loss_function,
    )
    logger.debug("Completed  training dataset Preprocess")

    AUTOTUNE = 1000
    # Update status to Training for map function in the preprocess
    update_status(True)

    # If input datatype is tfrecords or images
    if train_data.filetype != "tfrecords":
        t_path_ds = tf.data.Dataset.from_tensor_slices(train_data.files)
        t_image_ds = t_path_ds.map(format_example, num_parallel_calls=AUTOTUNE)
        t_label_ds = tf.data.Dataset.from_tensor_slices(
            tf.cast(train_data.labels, tf.int64)
        )
        t_image_label_ds = tf.data.Dataset.zip((t_image_ds, t_label_ds))
        # train_ds = t_image_label_ds.shuffle(buffer_size=train_data.min_images).repeat()
    else:
        t_path_ds = tf.data.TFRecordDataset(train_data.files)
        t_image_ds = t_path_ds.map(format_example_tf, num_parallel_calls=AUTOTUNE)
        # sys.exit(0)
        # min images variables should be update from number  of tfrecords to number of images
        num_image = 0
        for image, label in t_image_ds:
            num_image = num_image + 1
        # print(num_image)
        # sys.exit(0)
        train_data.min_images = num_image
        t_image_label_ds = tf.data.Dataset.zip(t_image_ds)
        # naresh:adding these additional steps to avoid shuffling on images and shuffle on imagepaths
        # t_image_ds = t_path_ds.shuffle(buffer_size=train_data.min_images).repeat()
        # train_ds = tf.data.Dataset.zip(t_image_ds)
        # train_ds = t_image_label_ds.shuffle(buffer_size=train_data.min_images).repeat()
    # train_ds = t_image_label_ds.shuffle(buffer_size=train_data.min_images)
    train_ds = t_image_label_ds
    # train_ds = t_image_label_ds.shuffle(buffer_size=train_data.min_images).repeat()
    train_ds = train_ds.repeat().batch(args.BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
    # train_ds = train_ds.batch(args.BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
    training_steps = int(train_data.min_images / args.BATCH_SIZE)
    logger.debug("Completed Training dataset")

    if args.image_dir_validation:
        # Get Validation data
        # Update status to Testing for map function in the preprocess
        update_status(False)
        validation_data = Preprocess(
            args.image_dir_validation,
            args.filetype,
            args.tfrecord_image,
            args.tfrecord_label,
            loss_function=args.loss_function,
        )
        logger.debug("Completed test dataset Preprocess")

        if validation_data.filetype != "tfrecords":
            v_path_ds = tf.data.Dataset.from_tensor_slices(validation_data.files)
            v_image_ds = v_path_ds.map(format_example, num_parallel_calls=AUTOTUNE)
            v_label_ds = tf.data.Dataset.from_tensor_slices(
                tf.cast(validation_data.labels, tf.int64)
            )
            v_image_label_ds = tf.data.Dataset.zip((v_image_ds, v_label_ds))
        else:
            v_path_ds = tf.data.TFRecordDataset(validation_data.files)
            v_image_ds = v_path_ds.map(format_example_tf, num_parallel_calls=AUTOTUNE)
            # min images variables should be update from number  of tfrecords to number of images
            num_image = 0
            for image, label in v_image_ds:
                num_image = num_image + 1
            validation_data.min_images = num_image
            v_image_label_ds = tf.data.Dataset.zip(v_image_ds)

        # validation_ds = v_image_label_ds.shuffle(buffer_size=validation_data.min_images)
        validation_ds = v_image_label_ds
        # num_img=0
        # for image, label in validation_ds:
        # print("Image shape: ", image.numpy().shape)
        # print("Label: ", label.numpy().shape)
        # print("Label: ", label.numpy())
        # num_img=num_img+1
        # print(num_img)
        # sys.exit(0)
        # validation_ds = validation_ds.batch(args.BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
        validation_ds = (
            validation_ds.repeat().batch(args.BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
        )

        validation_steps = int(validation_data.min_images / args.BATCH_SIZE)
        logger.debug("Completed Validation dataset")

    else:
        validation_ds = None
        validation_steps = None

    out_dir = os.path.join(
        args.log_dir,
        args.model_name
        + "_"
        + args.optimizer
        + "_"
        + str(args.lr)
        + "-"
        + args.loss_function,
    )
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    checkpoint_path = os.path.join(out_dir, "cp-{epoch:04d}.ckpt")
    checkpoint_dir = os.path.dirname(checkpoint_path)

    reg_drop_out_per = None
    if args.reg_drop_out_per is not None:
        reg_drop_out_per = float(args.reg_drop_out_per)

    l2_reg = None
    if args.l2_reg is not None:
        l2_reg = float(args.l2_reg)

    num_layers = None
    if args.train_num_layers is not None:
        num_layers = int(args.num_layers)

    logger.debug("Mirror initialized")
    GPU = True
    if GPU is True:
        # This must be fixed for multi-GPU
        mirrored_strategy = tf.distribute.MirroredStrategy()
        # mirrored_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
        # mirrored_strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
        with mirrored_strategy.scope():
            # if args.train_num_layers:
            m = GetModel(
                model_name=args.model_name,
                img_size=args.patch_size,
                classes=train_data.classes,
                num_layers=num_layers,
                reg_drop_out_per=reg_drop_out_per,
                l2_reg=l2_reg,
            )
            # else:
            #    m = GetModel(model_name=args.model_name, img_size=args.patch_size, classes=train_data.classes, reg_drop_out_per=reg_drop_out_per)
            # logger.debug('Model constructed')

            model = m.compile_model(args.optimizer, args.lr, args.loss_function)
            # print(model.summary())
            # sys.exit(0)
            # inside scope
            logger.debug("Model compiled")
            latest = tf.train.latest_checkpoint(checkpoint_dir)
            # print(latest)
            # sys.exit(0)
            if not latest:
                if args.prev_checkpoint:
                    model.load_weights(args.prev_checkpoint)
                    logger.debug("Loading weights from " + args.prev_checkpoint)
                model.save_weights(checkpoint_path.format(epoch=0))
                latest = tf.train.latest_checkpoint(checkpoint_dir)
            ini_epoch = int(re.findall(r"\b\d+\b", os.path.basename(latest))[0])
            print(latest, ini_epoch)

            logger.debug("Loading initialized model")
            model.load_weights(latest)
            # model.save(os.path.join(out_dir, 'my_model.h5'))
            # sys.exit(0)
            logger.debug("Loading weights from " + latest)

        logger.debug("Completed loading initialized model")
        cb = CallBacks(learning_rate=args.lr, log_dir=out_dir, optimizer=args.optimizer)
        logger.debug("Model image saved")  #
        model.fit(
            train_ds,
            steps_per_epoch=training_steps,
            epochs=args.num_epochs,
            callbacks=cb.get_callbacks(),
            validation_data=validation_ds,
            validation_steps=validation_steps,
            class_weight=None,
            max_queue_size=1000,
            workers=args.NUM_WORKERS,
            use_multiprocessing=args.use_multiprocessing,
            shuffle=False,
            initial_epoch=ini_epoch,
        )
        model.save(os.path.join(out_dir, "my_model.h5"))
    else:
        # if args.train_num_layers:
        #    m = GetModel(model_name=args.model_name, img_size=args.patch_size, classes=train_data.classes,
        #                 num_layers=int(args.train_num_layers))
        # else:
        #    m = GetModel(model_name=args.model_name, img_size=args.patch_size, classes=train_data.classes)
        m = GetModel(
            model_name=args.model_name,
            img_size=args.patch_size,
            classes=train_data.classes,
            num_layers=num_layers,
            reg_drop_out_per=reg_drop_out_per,
            l2_reg=l2_reg,
        )
        logger.debug("Model constructed")

        model = m.compile_model(args.optimizer, args.lr, args.loss_function)
        logger.debug("Model compiled")
        model.save_weights(checkpoint_path.format(epoch=0))
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        if not latest:
            model.save_weights(checkpoint_path.format(epoch=0))
            latest = tf.train.latest_checkpoint(checkpoint_dir)
        ini_epoch = int(re.findall(r"\b\d+\b", os.path.basename(latest))[0])
        logger.debug("Loading initialized model")
        model.load_weights(latest)
        logger.debug("Loading weights from " + latest)
        # if args.prev_checkpoint:
        # model.load_weights(args.prev_checkpoint)
        # logger.debug('Loading weights from ' + args.prev_checkpoint)
        logger.debug("Completed loading initialized model")
        cb = CallBacks(learning_rate=args.lr, log_dir=out_dir, optimizer=args.optimizer)
        logger.debug("Model image saved")
        model.fit(
            train_ds,
            steps_per_epoch=training_steps,
            epochs=args.num_epochs,
            callbacks=cb.get_callbacks(),
            validation_data=validation_ds,
            validation_steps=validation_steps,
            class_weight=None,
            max_queue_size=1000,
            workers=args.NUM_WORKERS,
            use_multiprocessing=args.use_multiprocessing,
            shuffle=False,
            initial_epoch=ini_epoch,
        )
        model.save(os.path.join(out_dir, "my_model.h5"))
