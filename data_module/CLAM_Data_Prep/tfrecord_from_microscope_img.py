import openslide
import tensorflow as tf
import os
import argparse
import sys
import numpy as np
from PIL import Image
import io
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt

"""function to check if input files exists and valid"""


def input_file_validity(file):
    """Validates the input files"""
    if os.path.exists(file) == False:
        raise argparse.ArgumentTypeError("\nERROR:Path:\n" + file + ":Does not exist")
    if os.path.isfile(file) == False:
        raise argparse.ArgumentTypeError(
            "\nERROR:File expected:\n" + file + ":is not a file"
        )
    if os.access(file, os.R_OK) == False:
        raise argparse.ArgumentTypeError("\nERROR:File:\n" + file + ":no read access ")
    return file


def argument_parse():
    """Parses the command line arguments"""
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-f", "--patch_file", help="Patch file", required="True")
    parser.add_argument("-s", "--sample", help="input file", required="True")
    parser.add_argument("-o", "--tf_output", help="output tf dir", required="True")
    return parser


"""creating binary mask to inspect areas with tissue and performance of threshold"""

"""code to extract feature vector"""


def patch_feature_extraction_resnet(input_shape=(512, 512, 3)):
    ## Load the ResNet50 model
    resnet50_model = tf.keras.applications.resnet50.ResNet50(
        include_top=False, weights="imagenet", input_shape=input_shape
    )

    resnet50_model.trainable = False  ## Free Training

    ## Create a new Model based on original resnet50 model ended after the 3rd residual block
    layer_name = "conv4_block1_0_conv"
    res50 = tf.keras.Model(
        inputs=resnet50_model.input, outputs=resnet50_model.get_layer(layer_name).output
    )

    ## Add adaptive mean-spatial pooling after the new model
    adaptive_mean_spatial_layer = tf.keras.layers.GlobalAvgPool2D()
    return res50, adaptive_mean_spatial_layer


def patch_feature_extraction(
    image_string, res50, adaptive_mean_spatial_layer, input_shape=(512, 512, 3)
):
    """
    Args:
        image_string:  bytes(PIL_image)
    :return: features:  Feature Vectors, float32
    """

    image_tensor = tf.io.decode_image(image_string)

    image_np = np.array(image_tensor)

    image_batch = np.expand_dims(image_np, axis=0)

    image_patch = tf.keras.applications.resnet50.preprocess_input(image_batch.copy())

    ## Return the feature vectors
    image_patch = tf.image.per_image_standardization(image_patch).numpy()
    predicts = res50.predict(image_patch)
    features = adaptive_mean_spatial_layer(predicts)
    features = tf.io.serialize_tensor(features)
    img_features = features.numpy()

    return img_features


"""TF2 helper functions for TF Records"""


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


"""TF2 helper functions for TF Records"""


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


"""TF2 helper functions for TF Records"""


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def create_tfrecord_img(patch_file, sample, tf_output):
    writer = tf.io.TFRecordWriter(os.path.join(tf_output, sample + ".tfrecords"))
    fobj = open(patch_file)
    res50, adaptive_mean_spatial_layer = patch_feature_extraction_resnet()
    for file in fobj:
        file = file.strip()
        print(file)
        img = Image.open(file)
        img = img.convert("RGB")

        image_name = os.path.basename(file)
        mut_type = int(os.path.basename(os.path.dirname(file)))

        image_string = open(file, "rb").read()

        patch_size = 256
        image_feature = patch_feature_extraction(
            image_string,
            res50,
            adaptive_mean_spatial_layer,
            (patch_size, patch_size, 3),
        )
        image_format = "jpeg"
        """writing tfrecord"""
        feature = {
            "height": _int64_feature(patch_size),
            "width": _int64_feature(patch_size),
            "depth": _int64_feature(3),
            "label": _int64_feature(mut_type),
            "image/format": _bytes_feature(image_format.encode("utf8")),
            "image_name": _bytes_feature(image_name.encode("utf8")),
            "image/encoded": _bytes_feature(image_string),
            "image_feature": _bytes_feature(image_feature),
        }

        Example = tf.train.Example(features=tf.train.Features(feature=feature))
        Serialized = Example.SerializeToString()
        writer.write(Serialized)
    fobj.close()
