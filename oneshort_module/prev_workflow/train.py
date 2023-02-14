from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import logging
import os
import sys
import tensorflow as tf
from callbacks import CallBacks

# from model_factory import GetModel, build_triplet_model
from preprocess import (
    Preprocess,
    format_example,
    format_example_tf,
    update_status,
    create_triplets_oneshot,
    create_triplets_oneshot_img_v,
)
from preprocess import create_triplets_oneshot_img
from data_runner import DataRunner
from steps import write_tb
import numpy as np
from sklearn import metrics
import re
from sklearn.metrics import roc_curve, roc_auc_score
from PIL import Image, ImageDraw
from losses import triplet_loss as loss_fn
from model_factory import GetModel
from tensorflow.keras import models
from PIL import Image, ImageDraw

# os.environ['CUDA_VISIBLE_DEVICES']="2,3"
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))
# if len(tf.config.experimental.list_physical_devices('GPU')) == 0:
#    exit()
# for g in tf.config.list_physical_devices('GPU'):
#    tf.config.experimental.set_memory_growth(g, True)

tf.config.set_soft_device_placement(True)

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))
# if len(tf.config.experimental.list_physical_devices('GPU')) == 0:
#    exit()

tf.config.set_soft_device_placement(True)
# tf.debugging.set_log_device_placement(True)
###############################################################################
# Input Arguments
###############################################################################

parser = argparse.ArgumentParser(
    description="Run a Siamese Network with a triplet loss on a folder of images."
)
parser.add_argument(
    "-t",
    "--image_dir_train",
    dest="image_dir_train",
    required=True,
    help="File path ending in folders that are to be used for model training",
)

parser.add_argument(
    "-v",
    "--image_dir_validation",
    dest="image_dir_validation",
    default=None,
    help="File path ending in folders that are to be used for model validation",
)

parser.add_argument(
    "-m",
    "--model-name",
    dest="model_name",
    default="custom",
    choices=[
        "custom",
        "DenseNet121",
        "DenseNet169",
        "DenseNet201",
        "InceptionResNetV2",
        "InceptionV3",
        "MobileNet",
        "MobileNetV2",
        "NASNetLarge",
        "NASNetMobile",
        "ResNet50",
        "ResNet152",
        "VGG16",
        "VGG19",
        "Xception",
    ],
    help="Models available from tf.keras",
)

parser.add_argument(
    "-o",
    "--optimizer-name",
    dest="optimizer",
    default="Adam",
    choices=[
        "Adadelta",
        "Adagrad",
        "Adam",
        "Adamax",
        "Ftrl",
        "Nadam",
        "RMSprop",
        "SGD",
    ],
    help="Optimizers from tf.keras",
)

parser.add_argument(
    "-p",
    "--patch_size",
    dest="patch_size",
    help="Patch size to use for training",
    default=256,
    type=int,
)

parser.add_argument(
    "-c",
    "--embedding_size",
    dest="embedding_size",
    help="How large should the embedding dimension be",
    default=128,
    type=int,
)

parser.add_argument(
    "-l",
    "--log_dir",
    dest="log_dir",
    default="log_dir",
    help="Place to store the tensorboard logs",
)

parser.add_argument(
    "-L",
    "--nb_layers",
    dest="nb_layers",
    default=99,
    type=int,
    help="Maximum number of layers to train in the model",
)

parser.add_argument(
    "-r", "--learning-rate", dest="lr", help="Learning rate", default=0.01, type=float
)

parser.add_argument(
    "-e",
    "--num-epochs",
    dest="num_epochs",
    help="Number of epochs to use for training",
    default=5,
    type=int,
)

parser.add_argument(
    "-b",
    "--batch-size",
    dest="BATCH_SIZE",
    help="Number of batches to use for training",
    default=1,
    type=int,
)

parser.add_argument(
    "-V",
    "--verbose",
    dest="logLevel",
    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    default="DEBUG",
    help="Set the logging level",
)

parser.add_argument(
    "-F",
    "--filetype",
    dest="filetype",
    choices=["tfrecords", "images"],
    default="images",
    help="Set the logging level",
)

parser.add_argument(
    "--tfrecord_image",
    dest="tfrecord_image",
    default="image/encoded",
    help="Set the logging level",
)

parser.add_argument(
    "--tfrecord_label",
    dest="tfrecord_label",
    default="null",
    help="Set the logging level",
)

parser.add_argument(
    "-f",
    "--log_freq",
    dest="log_freq",
    default=100,
    help="Set the logging frequency for saving Tensorboard updates",
    type=int,
)

parser.add_argument(
    "-a",
    "--accuracy_num_batch",
    dest="acc_num_batch",
    default=20,
    help="Number of batches to consider to calculate training and validation accuracy",
    type=int,
)

args = parser.parse_args()

logging.basicConfig(
    stream=sys.stderr,
    level=args.logLevel,
    format="%(name)s (%(levelname)s): %(message)s",
)

logger = logging.getLogger(__name__)
logger.setLevel(args.logLevel)

###############################################################################
# Begin priming the data generation pipeline
###############################################################################

# Get Training and Validation data
train_data = Preprocess(
    args.image_dir_train, args.filetype, args.tfrecord_image, args.tfrecord_label
)
logger.debug("Completed  training dataset Preprocess")

# AUTOTUNE = 1000
AUTOTUNE = tf.data.experimental.AUTOTUNE
# Update status to Training for map function in the preprocess
update_status(True)

# If input datatype is tfrecords or images
if train_data.filetype != "tfrecords":
    t_path_ds = tf.data.Dataset.from_tensor_slices(train_data.files)
    t_image_ds = t_path_ds.map(format_example, num_parallel_calls=AUTOTUNE)
    t_label_ds = tf.data.Dataset.from_tensor_slices(train_data.labels)
    # t_image_label_ds, train_data.min_images, train_image_labels = create_triplets_oneshot_img(t_image_ds, t_label_ds, args.patch_size)
    (
        t_image_label_ds,
        train_data.min_images,
        train_image_labels,
    ) = create_triplets_oneshot_img_v(t_image_ds, t_label_ds)
else:
    t_path_ds = tf.data.TFRecordDataset(train_data.files)
    t_image_ds = t_path_ds.map(format_example_tf, num_parallel_calls=AUTOTUNE)
    t_image_label_ds, train_data.min_images = create_triplets_oneshot(t_image_ds)

train_ds_dr = DataRunner(t_image_label_ds)
logger.debug("Completed Data runner")

train_ds = tf.data.Dataset.from_generator(
    train_ds_dr.get_distributed_datasets,
    output_types=(
        {
            "anchor_img": tf.float32,
            "other_img": tf.float32,
        },
        tf.int64,
    ),
    output_shapes=(
        {
            "anchor_img": [args.patch_size, args.patch_size, 3],
            "other_img": [args.patch_size, args.patch_size, 3],
        },
        (2,),
    ),
)

# num_img=0
# for image, label in train_ds:
# if num_img<3:
# npa=image["anchor_img"].numpy()
# im = Image.fromarray(np.uint8(npa*255))
# im.save(str(num_img) + '_anchor.png', "png")
# npa = image["other_img"].numpy()
# im = Image.fromarray(np.uint8(npa * 255))
# im.save(str(num_img) + '_other.png', "png")
# print("Image shape: ", image["anchor_img"].numpy().shape)
# print("Image shape: ", image["other_img"].numpy().shape)
# print("Label: ", label.numpy().shape)
# print("Label: ", label.numpy())
# num_img=num_img+1
# print(num_img)
# sys.exit(0)
train_data_num = 0
for img_data, labels in train_ds:
    train_data_num = train_data_num + 1
training_steps = int(train_data_num / args.BATCH_SIZE)
# train_ds = train_ds.shuffle(train_data_num, reshuffle_each_iteration=True).repeat().batch(args.BATCH_SIZE, drop_remainder=True)
train_ds = train_ds.repeat().batch(args.BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)


# train_ds = train_ds.shuffle(buffer_size=train_data_num).repeat()
# train_ds = train_ds.batch(args.BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
# train_ds =train_ds.batch(args.BATCH_SIZE, drop_remainder=True)
# print(train_data_num,args.BATCH_SIZE,training_steps)
# sys.exit(0)
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
    )
    logger.debug("Completed test dataset Preprocess")

    if validation_data.filetype != "tfrecords":
        v_path_ds = tf.data.Dataset.from_tensor_slices(validation_data.files)
        v_image_ds = v_path_ds.map(format_example, num_parallel_calls=AUTOTUNE)
        v_label_ds = tf.data.Dataset.from_tensor_slices(validation_data.labels)
        (
            v_image_label_ds,
            validation_data.min_images,
            validation_image_labels,
        ) = create_triplets_oneshot_img_v(v_image_ds, v_label_ds)
    else:
        v_path_ds = tf.data.TFRecordDataset(validation_data.files)
        v_image_ds = v_path_ds.map(format_example_tf, num_parallel_calls=AUTOTUNE)
        v_image_label_ds, validation_data.min_images = create_triplets_oneshot(
            v_image_ds
        )
    v_ds_dr = DataRunner(v_image_label_ds)
    logger.debug("Completed Data runner")
    validation_ds = tf.data.Dataset.from_generator(
        v_ds_dr.get_distributed_datasets,
        output_types=(
            {
                "anchor_img": tf.float32,
                "other_img": tf.float32,
            },
            tf.int64,
        ),
        output_shapes=(
            {
                "anchor_img": [args.patch_size, args.patch_size, 3],
                "other_img": [args.patch_size, args.patch_size, 3],
            },
            (2,),
        ),
    )
    # validation_ds = validation_ds.batch(args.BATCH_SIZE)
    # validation_steps = int(validation_data.min_images / args.BATCH_SIZE)
    #
    # num_img = 0
    # for image, label in validation_ds:
    #     if num_img < 10:
    #         npa = image["anchor_img"].numpy()
    #         im = Image.fromarray(np.uint8(npa * 255))
    #         im.save(str(num_img) + '_anchor.png', "png")
    #         npa = image["other_img"].numpy()
    #         im = Image.fromarray(np.uint8(npa * 255))
    #         im.save(str(num_img) + '_other.png', "png")
    #         print("Image shape: ", image["anchor_img"].numpy().shape)
    #         print("Image shape: ", image["other_img"].numpy().shape)
    #         print("Label: ", label.numpy().shape)
    #         print("Label: ", label.numpy())
    #         num_img = num_img + 1
    # print(num_img)

    validation_data_num = 0
    for img_data, label in validation_ds:
        # print("Label: ", label.numpy().shape)
        # print("Label: ", label.numpy())
        validation_data_num = validation_data_num + 1
    validation_steps = int(validation_data_num / args.BATCH_SIZE)
    # validation_ds = validation_ds.shuffle(validation_data_num, reshuffle_each_iteration=True).repeat().batch(args.BATCH_SIZE, drop_remainder=True)
    # validation_ds = validation_ds.shuffle(validation_data_num, reshuffle_each_iteration=True).batch(args.BATCH_SIZE, drop_remainder=True)
    # validation_ds = validation_ds.shuffle(buffer_size=validation_data_num).repeat().batch(args.BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
    validation_ds = (
        validation_ds.repeat().batch(args.BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
    )

    # validation_ds =validation_ds.batch(args.BATCH_SIZE, drop_remainder=True)
    # validation_data_num = validation_ds.shuffle(buffer_size=validation_data_num).repeat()
    # validation_data_num = validation_data_num.batch(args.BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

    logger.debug("Completed Validation dataset")
    # sys.exit(0)
else:
    validation_ds = None
    validation_steps = None


# ####################################################################
# Temporary cleaning function
# ####################################################################
out_dir = os.path.join(
    args.log_dir,
    args.model_name
    + "_"
    + args.optimizer
    + "_"
    + str(args.lr)
    + "_"
    + str(args.embedding_size),
)
checkpoint_name = "training_checkpoints"


###############################################################################
# Define callbacks
###############################################################################
cb = CallBacks(learning_rate=args.lr, log_dir=out_dir, optimizer=args.optimizer)
###############################################################################
# Define callbacks
###############################################################################
# def scheduler(epoch):
#     if epoch < 5:
#         #return 0.01
#         return 0.0006
#     elif epoch < 10:
#         #return 0.001
#         return 0.0006
#     else:
#         #return 0.001 * tf.math.exp(0.1 * (10 - epoch))
#         return 0.0006
#
# cb = [tf.keras.callbacks.TensorBoard(log_dir=out_dir, histogram_freq=1, write_graph=True, update_freq='epoch',write_images=False),
#       tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(out_dir, 'cp-{epoch:04d}.ckpt'), monitor='mse', verbose=1,save_weights_only=True,save_frequency=1,
#                                          mode='auto'),
#         tf.keras.callbacks.EarlyStopping(monitor='loss', patience=30)]
# tf.keras.callbacks.LearningRateScheduler(scheduler)]

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

checkpoint_path = os.path.join(out_dir, "cp-{epoch:04d}.ckpt")
checkpoint_dir = os.path.dirname(checkpoint_path)

###############################################################################
# Build model
###############################################################################
training_flag = 1
if training_flag == 1:
    ###############################################################################
    # Build model
    ###############################################################################
    traditional = True
    if traditional is True:
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            m = GetModel(
                model_name=args.model_name,
                img_size=args.patch_size,
                embedding_size=args.embedding_size,
            )
            model = m.build_model()
            model.summary()
            # print(args.lr)
            # sys.exit(0)
            optimizer = m.get_optimizer(args.optimizer, lr=args.lr)
            model.compile(
                optimizer=optimizer,
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[
                    tf.keras.metrics.BinaryCrossentropy(name="loss"),
                    tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                ],
            )
            ## tf.keras.metrics.AUC(curve='PR', num_thresholds=10, name='PR'),# metrics=['binary_accuracy', 'mse', tf.keras.metrics.AUC()]
            # tf.keras.metrics.Accuracy(name='accuracy'),
            # tf.keras.metrics.CategoricalAccuracy(name='CategoricalAccuracy'),
            logger.debug("Model compiled")
            latest = tf.train.latest_checkpoint(checkpoint_dir)
            if not latest:
                model.save_weights(checkpoint_path.format(epoch=0))
                latest = tf.train.latest_checkpoint(checkpoint_dir)
            ini_epoch = int(re.findall(r"\b\d+\b", os.path.basename(latest))[0])
            logger.debug("Loading initialized model")
            model.load_weights(latest)
            logger.debug("Loading weights from " + latest)

        logger.debug("Completed loading initialized model")

        # if args.image_dir_validation is None:
        # model.fit(train_ds, epochs=args.num_epochs, callbacks=cb)
        # else:
        # model.fit(train_ds, epochs=args.num_epochs, callbacks=cb, validation_data=validation_ds,steps_per_epoch=training_steps,)
        model.fit(
            train_ds,
            epochs=args.num_epochs,
            callbacks=cb.get_callbacks(),
            validation_data=validation_ds,
            steps_per_epoch=training_steps,
            validation_steps=validation_steps,
            workers=1,
            class_weight=None,
            max_queue_size=1000,
            use_multiprocessing=False,
            shuffle=False,
            initial_epoch=ini_epoch,
        )
        # steps_per_epoch=training_steps,
        # epochs=args.num_epochs,
        # callbacks=cb.get_callbacks(),
        # validation_data=validation_ds,
        # validation_steps=validation_steps)

        # outfile_dir = os.path.join(out_dir, 'siamesenet')
        # model.reset_metrics()
        # model.save(outfile_dir, save_format='tf')
        # os.makedirs(outfile_dir)
        model.save(os.path.join(out_dir, "my_model.h5"))
        # model.save(outfile_dir, 'my_model.h5')
        # print('Completed and saved {outfile_dir}')
    else:
        # strategy = tf.distribute.MirroredStrategy()
        # with strategy.scope():
        m = GetModel(
            model_name=args.model_name,
            img_size=args.patch_size,
            embedding_size=args.embedding_size,
        )
        logger.debug("Model constructed")
        model = m.build_model()
        model.summary()
        logger.debug("Model built")
        optimizer = m.get_optimizer(args.optimizer, lr=args.lr)
        writer = tf.summary.create_file_writer(out_dir)
        for epoc in range(1, args.num_epochs + 1):
            for step, (x_batch_train, y_batch_train) in enumerate(train_ds):
                step *= epoc
                with tf.GradientTape() as tape:
                    logits = model(x_batch_train, training=True)
                    loss_value = loss_fn(y_batch_train, logits)
                grads = tape.gradient(loss_value, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                if step % args.log_freq == 0:
                    print(f"\rStep: {step}\tLoss: {loss_value[0]:04f}", end="")
                    with writer.as_default():
                        tf.summary.scalar("dist", loss_value[0], step=step)

else:
    model = models.load_model(os.path.join(out_dir, "my_model.h5"))
    # imported = tf.saved_model.load(out_dir)
    # infer = imported.signatures["serving_default"]
    # print(list(imported.signatures.keys()))
    # sys.exit(0)
    # sys.exit(0)
    # m = GetModel(model_name=args.model_name, img_size=args.patch_size, embedding_size=128)
    # logger.debug('Model constructed')
    # model = m.build_model()
    # logger.debug('Model built')

    # model = m.compile_model(args.optimizer, args.lr, img_size=args.patch_size)
    # logger.debug('Model compiled')
    # latest = tf.train.latest_checkpoint(out_dir+'/'+variables)
    # print(latest)
    # sys.exit(0)
    # model.load_weights(latest)

    for img_data, labels in train_ds:
        # img_data, labels = data
        lab = labels.numpy().tolist()
        # print(img_data[0].numpy().shape)
        pos_img, neg_img = img_data["anchor_img"], img_data["other_img"]
        result = np.asarray(model.predict([pos_img, neg_img])).tolist()
        for i in range(len(lab)):
            print("train", lab[i][0], result[i][0])
    # sys.exit(0)
    for img_data, labels in validation_ds:
        # img_data, labels = data
        lab = labels.numpy().tolist()
        # print(img_data[0].numpy().shape)
        pos_img, neg_img = img_data["anchor_img"], img_data["other_img"]
        result = np.asarray(model.predict([pos_img, neg_img])).tolist()
        for i in range(len(lab)):
            print("valid", lab[i][0], result[i][0])
