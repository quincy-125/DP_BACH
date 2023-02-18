import os
import logging
import tensorflow as tf
import random
from model_factory import get_emb_vec

logger = logging.getLogger(__name__)
global tf_image, tf_label, status
import numpy as np
import sys


class Preprocess:
    def __init__(self, directory_path, filetype, tfrecord_image, tfrecord_label):
        """
        Return a randomized list of each directory's contents

        :param directory_path: a directory that contains sub-folders of images

        :returns class_files: a dict of each file in each folder
        """
        global tf_image, tf_label
        tf_image = tfrecord_image
        tf_label = tfrecord_label
        logger.debug("Initializing Preprocess")
        self.directory_path = directory_path
        self.filetype = filetype
        self.classes = self.__get_classes()
        self.tfrecord_image = tfrecord_image
        self.tfrecord_label = tfrecord_label
        self.files1, self.files2, self.labels, self.num = self.__get_lists()

    def __check_min_image(self, prev, new):
        logger.debug("Counting the number of images")
        # if prev is None or prev > new:
        # return new
        # else:
        # return prev
        if prev is None:
            return new
        else:
            return prev + new

    def __get_classes(self):
        classes = os.listdir(self.directory_path)
        return classes.__len__()

    def __get_lists(self):
        logging.debug("Getting initial list of images and labels")

        files = []
        labels = []
        label_dict = dict()
        label_number = 0
        min_images = None
        filetype = self.filetype
        tfrecord_image = self.tfrecord_image
        tfrecord_label = self.tfrecord_label
        classes = os.listdir(self.directory_path)
        for x in classes:
            class_files = os.listdir(os.path.join(self.directory_path, x))
            class_files = [os.path.join(self.directory_path, x, j) for j in class_files]
            class_labels = [label_number for x in range(class_files.__len__())]
            min_images = self.__check_min_image(min_images, class_labels.__len__())
            label_dict[x] = label_number
            label_number += 1
            files.extend(class_files)
            labels.extend(class_labels)

        # labels = tf.dtypes.cast(labels, tf.uint8)

        list_images = files
        list_labels = labels

        # unique labels
        unique_labels = list(set(list_labels))
        unique_labels_num = []
        unique_labels_index = []
        # creating array of indexes per label and number of images per label
        for label in unique_labels:
            inx = [x for x in range(0, len(list_labels)) if list_labels[x] == label]
            unique_labels_num.append(len(inx))
            unique_labels_index.append(inx)

        # max number of images per label category
        max_unique_labels_num = max(unique_labels_num)

        # randomly selecting images for a,p & n class
        list_img_index1 = []
        list_img_index2 = []
        list_img_label = []
        num = 0
        # iterating through all classes
        for i in range(0, len(unique_labels)):
            # iterating number of times equal to max image count of all category (doing this step , not to over represent
            # the categories with more images)
            for j in range(0, max_unique_labels_num):
                tmp_a_idx = i
                tmp_p_idx = tmp_a_idx
                tmp_unique_labels = list(range(0, len(unique_labels)))
                tmp_unique_labels.remove(tmp_p_idx)
                # selecting other category for 'n' class
                tmp_n_idx = random.choices(tmp_unique_labels, k=1)[0]
                # selecting image index for 'a','n' & 'p' category randonly
                tmp_a_idx_img = random.choices(unique_labels_index[tmp_a_idx], k=1)[0]
                tmp_p_idx_img = random.choices(unique_labels_index[tmp_p_idx], k=1)[0]
                tmp_n_idx_img = random.choices(unique_labels_index[tmp_n_idx], k=1)[0]
                # extracting actual images with selected indexes for each class 'a','p' & 'n'
                # list_img_index.append((list_images[tmp_a_idx_img], list_images[tmp_p_idx_img], list_images[tmp_n_idx_img]))
                # list_img_label.append([tmp_a_idx, tmp_p_idx, tmp_n_idx])
                # list_img_index.append((list_images[tmp_a_idx_img], list_images[tmp_p_idx_img], [0,1]))
                list_img_index1.append(list_images[tmp_a_idx_img])
                list_img_index2.append(list_images[tmp_p_idx_img])
                list_img_label.append(0)
                # list_img_label.append([tmp_a_idx, tmp_p_idx])
                # list_img_index.append((list_images[tmp_a_idx_img], list_images[tmp_n_idx_img], [1,0]))
                # list_img_label.append([tmp_a_idx, tmp_n_idx])
                list_img_index1.append(list_images[tmp_a_idx_img])
                list_img_index2.append(list_images[tmp_n_idx_img])
                list_img_label.append(1)
                num = num + 2
        idx = list(range(0, len(list_img_index1)))
        random.shuffle(idx)
        random.shuffle(idx)
        list_img_index1 = [list_img_index1[i] for i in idx]
        list_img_index2 = [list_img_index2[i] for i in idx]
        list_img_label = [list_img_label[i] for i in idx]
        labels = tf.one_hot(list_img_label, 2)
        # return list_img_index1, max_unique_labels_num, list_img_label1

        # return files, labels, label_dict, min_images
        return list_img_index1, list_img_index2, labels, num


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
        image = tf.image.random_brightness(image, max_delta=0.02, seed=44)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_hue(image, max_delta=0.02)
    # image = tf.image.per_image_standardization(image)
    image = tf.reshape(image, (img_size, img_size, 3))

    return image
