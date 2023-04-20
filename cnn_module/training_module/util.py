from random import shuffle, choice
import os
import logging
import tensorflow as tf
import sys
from PIL import Image

logger = logging.getLogger(__name__)
global tf_image, tf_label, status
import random
import numpy as np


# from skimage.color import rgb2lab,rgb2hed
class Preprocess:
    def __init__(
        self,
        args,
        directory_path,
    ):
        """
        Return a randomized list of each directory's contents

        :param directory_path: a directory that contains sub-folders of images

        :returns class_files: a dict of each file in each folder
        """
        self.args = args
        # self.tfrecord_image = self.args.tfrecord_image
        # self.tfrecord_label = self.args.tfrecord_label
        self.directory_path = directory_path

        (
            self.files,
            self.labels,
            self.label_dict,
            self.min_images,
            self.filetype,
            self.tfrecord_image,
            self.tfrecord_label,
        ) = self.__get_lists()

    def __check_min_image(self, prev, new):
        logger.debug("Counting the number of images")
        if prev is None:
            return new
        else:
            return prev + new

    def __get_lists(self):
        logging.debug("Getting initial list of images and labels")
        # modified code to handle thousands of images with shuffling at file names instead of tf.shuffle
        files = list()
        labels = list()
        files_list = list()
        labels_list = list()
        len_list = list()
        label_dict = dict()

        label_number = 0
        min_images = None

        # tfrecord_image = self.tfrecord_image
        # tfrecord_label = self.tfrecord_label
        # classes = sorted(os.listdir(self.directory_path))
        flag = 1
        for x in self.args.classes:
            class_files = os.listdir(os.path.join(self.directory_path, x))
            class_files = [os.path.join(self.directory_path, x, j) for j in class_files]
            if ".tfrecord" in class_files[0]:
                flag = 0
            class_labels = [label_number for x in range(class_files.__len__())]
            min_images = self.__check_min_image(min_images, class_labels.__len__())
            label_dict[x] = label_number
            label_number += 1
            if flag == 0:
                files = files + class_files
                labels = labels + class_labels
            else:
                files_list.append(class_files)
                labels_list.append(class_labels)
                len_list.append(len(class_files))

        if flag == 0:
            """shuffling images"""
            idx = list(range(0, len(files)))
            random.shuffle(idx)
            files = [files[i] for i in idx]
            labels = [labels[i] for i in idx]
        else:
            max_len = max(len_list)
            files = [None] * self.args.classes * max_len
            labels = [None] * self.args.classes * max_len
            for i in range(0, self.args.classes, 1):
                tmp = files_list[i]

                if max_len > len(tmp):
                    tmp1 = tmp + [
                        tmp[z]
                        for z in np.random.randint(
                            len(tmp), size=(max_len - len(tmp))
                        ).tolist()
                    ]
                else:
                    tmp1 = tmp
                files[i :: self.args.classes] = tmp1
                labels[i :: self.args.classes] = [labels_list[i][0]] * len(tmp1)
            del tmp1
            del tmp
        del files_list
        del labels_list
        labels = tf.dtypes.cast(labels, tf.uint8)
        # I noticed that if your loss function expects loss, it has to be one hot, otherwise, it expects an int
        if not self.loss_function.startswith("Sparse"):
            labels = tf.one_hot(labels, self.args.classes)
        return (
            files,
            labels,
            label_dict,
            min_images,
            self.args.filetype,
            self.args.tfrecord_image,
            self.args.tfrecord_label,
        )


def update_status(stat):
    global status
    status = stat
    return stat


# processing images
def format_example(image_name=None, img_size=256):
    """
    Apply any image preprocessing here
    :param image_name: the specific filename of the image
    :param img_size: size that images should be reshaped to
    :param train: whether this is for training or not

    :return: image
    """
    global status
    train = status
    image = tf.io.read_file(image_name)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32) / 255
    image = tf.image.resize(image, (img_size, img_size))
    if train is True:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)

        image = tf.image.random_brightness(image, max_delta=0.2, seed=44)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5, seed=44)
        image = tf.clip_by_value(image, 0.0, 1.0)
    image = tf.reshape(image, (img_size, img_size, 3))

    return image


# extracting images and labels from tfrecords
def format_example_tf(tfrecord_proto, img_size=256):
    # Parse the input tf.Example proto using the dictionary above.
    # Create a dictionary describing the features.
    global tf_image, tf_label, status
    train = status
    image_feature_description = {
        tf_image: tf.io.FixedLenFeature((), tf.string, ""),
        tf_label: tf.io.FixedLenFeature((), tf.int64, -1),
    }
    parsed_image_dataset = tf.io.parse_single_example(
        tfrecord_proto, image_feature_description
    )
    image = parsed_image_dataset[tf_image]
    label = parsed_image_dataset[tf_label]
    label = tf.dtypes.cast(label, tf.uint8)
    label = tf.one_hot(label, 2)
    image = tf.io.decode_png(image, channels=3)
    image = tf.cast(image, tf.float32)  # /255
    image = (image / 127.5) - 1
    image = tf.image.resize(image, (img_size, img_size))

    if train is True:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_hue(image, 0.08)
        image = tf.image.random_saturation(image, 0.6, 1.6)
        image = tf.image.random_brightness(image, 0.05)
        image = tf.image.random_contrast(image, 0.7, 1.3)

    image = tf.reshape(image, (img_size, img_size, 3))
    return image, label
