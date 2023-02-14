import os
import logging
import tensorflow as tf
import random
from model_factory import get_emb_vec

logger = logging.getLogger(__name__)
global tf_image, tf_label, status
import numpy as np


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

        labels = tf.dtypes.cast(labels, tf.uint8)
        return (
            files,
            labels,
            label_dict,
            min_images,
            filetype,
            tfrecord_image,
            tfrecord_label,
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
    :return: image
    """
    global status
    train = status
    # image = tf.io.read_file(image_name)
    # image = tf.io.decode_jpeg(image, channels=3)
    # image = tf.cast(image, tf.float32)#/255
    # image = (image/127.5) - 1
    # #image = tf.image.per_image_standardization(image)
    # image = tf.image.resize(image, (img_size, img_size))

    # if train is True:
    # image = tf.image.random_flip_left_right(image)
    # #image = tf.image.random_brightness(image, 0.4)
    # #image = tf.image.random_contrast(image, lower=0.0, upper=0.1)
    # image = tf.image.random_flip_up_down(image)
    # #image = tf.image.random_hue(image, max_delta=0.2)
    # image = tf.image.random_hue(image, 0.08)
    # image = tf.image.random_saturation(image, 0.6, 1.6)
    # image = tf.image.random_brightness(image, 0.05)
    # image = tf.image.random_contrast(image, 0.7, 1.3)
    # #image = tf.image.rot90(image, tf.random_uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    # image = tf.reshape(image, (img_size, img_size, 3))

    image = tf.io.read_file(image_name)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32) / 255
    image = tf.image.resize(image, (img_size, img_size))
    if train is True:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.02, seed=44)
        # image = tf.image.random_contrast(image,lower=0.01, upper=0.02, seed=43)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_hue(image, max_delta=0.02)
    # image = tf.image.per_image_standardization(image)
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
    # image = parsed_image_dataset['image/encoded']
    # label = parsed_image_dataset['phenotype/TP53_Mutations']
    image = parsed_image_dataset[tf_image]
    label = parsed_image_dataset[tf_label]
    label = tf.dtypes.cast(label, tf.uint8)
    # label = tf.one_hot(label, 2)
    image = tf.io.decode_png(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)
    image = tf.image.resize(image, (img_size, img_size))

    if train is True:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.0, upper=0.1)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_hue(image, max_delta=0.2)

    # image = tf.reshape(image, (img_size, img_size, 3))
    return image, label


# Create pairs for one shot learning
def create_triplets_oneshot(t_image_ds):
    list_images = []
    list_labels = []
    # adding images and labels to the list
    for image, label in t_image_ds:
        list_images.append(image)
        list_labels.append(label.numpy())
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
    max_unique_labels_num = max(unique_labels_num) * len(unique_labels)

    # randomly selecting images for a,p & n class
    list_img_index = []
    list_img_label = []
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
            list_img_index.append(
                (list_images[tmp_a_idx_img], list_images[tmp_p_idx_img], [0, 1])
            )
            list_img_label.append([tmp_a_idx, tmp_p_idx])
            list_img_index.append(
                (list_images[tmp_a_idx_img], list_images[tmp_n_idx_img], [1, 0])
            )
            list_img_label.append([tmp_a_idx, tmp_n_idx])
    return list_img_index, max_unique_labels_num, list_img_label


def create_triplets_oneshot_img_v(t_image_ds, t_label_ds):
    """
    Args:
        t_image_ds: image list
        t_label_ds: image class
    Returns:
            list_img_index: a set of images
            max_unique_labels_num: number of images that will be in a batch (so we dont over represent one class)
            list_img_label: a label for each of the three images in a triplet (e.g. [0, 0, 1])
    """
    list_images = []
    list_labels = []
    # adding images and labels to the list
    for image in t_image_ds:
        list_images.append(image)

    for label in t_label_ds:
        list_labels.append(label.numpy())
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
    list_img_index = []
    list_img_label = []
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
            list_img_index.append(
                (list_images[tmp_a_idx_img], list_images[tmp_p_idx_img], [0, 1])
            )
            list_img_label.append([tmp_a_idx, tmp_p_idx])
            list_img_index.append(
                (list_images[tmp_a_idx_img], list_images[tmp_n_idx_img], [1, 0])
            )
            list_img_label.append([tmp_a_idx, tmp_n_idx])
    idx = list(range(0, len(list_img_index)))
    random.shuffle(idx)
    random.shuffle(idx)
    list_img_index1 = [list_img_index[i] for i in idx]
    del list_img_index
    list_img_label1 = [list_img_label[i] for i in idx]
    del list_img_label, idx
    return list_img_index1, max_unique_labels_num, list_img_label1


def create_triplets_oneshot_img(t_image_ds, t_label_ds):
    """

    Args:
        t_image_ds: image list
        t_label_ds: image class

    Returns:
            list_img_index: a set of images
            max_unique_labels_num: number of images that will be in a batch (so we dont over represent one class)
            list_img_label: a label for each of the three images in a triplet (e.g. [0, 0, 1])
    """
    # get model
    patch_size = 224
    img_size = 224
    IMG_SHAPE = (patch_size, patch_size, 3)
    IMG_SHAPE_NEW = (1, patch_size, patch_size, 3)
    conv_model = get_emb_vec(IMG_SHAPE)
    # model = conv_model.build_model()

    list_images = []
    list_labels = []
    # adding images and labels to the list
    for image in t_image_ds:
        list_images.append(image)

    for label in t_label_ds:
        list_labels.append(label.numpy())
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

    batch_loss_dist = []
    # iterating through all classes
    for i in range(0, len(unique_labels)):
        # iterating number of times equal to max image count of all category (doing this step , not to over represent
        # the categories with more images)
        for j in range(0, int(max_unique_labels_num / 10)):
            # print(i,j,max_unique_labels_num)
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
            A = conv_model.predict(
                tf.reshape(
                    tf.image.resize(list_images[tmp_a_idx_img], (img_size, img_size)),
                    IMG_SHAPE_NEW,
                )
            )
            P = conv_model.predict(
                tf.reshape(
                    tf.image.resize(list_images[tmp_p_idx_img], (img_size, img_size)),
                    IMG_SHAPE_NEW,
                )
            )
            N = conv_model.predict(
                tf.reshape(
                    tf.image.resize(list_images[tmp_n_idx_img], (img_size, img_size)),
                    IMG_SHAPE_NEW,
                )
            )
            studybatchloss = np.sum(np.square(A - P), axis=1) - np.sum(
                np.square(A - N), axis=1
            )
            batch_loss_dist.append(studybatchloss[0])
    batch_loss_dist_25 = np.percentile(batch_loss_dist, 25)
    batch_loss_dist_75 = np.percentile(batch_loss_dist, 75)
    x = [i for i in batch_loss_dist if i < batch_loss_dist_25]
    y = [i for i in batch_loss_dist if i > batch_loss_dist_75]
    # print(len(batch_loss_dist),len(x),len(y))
    print(batch_loss_dist_25, batch_loss_dist_75)
    # exit(0)
    # *len(unique_labels)
    # randomly selecting images for a,p & n class
    list_img_index = []
    list_img_label = []

    num_limit = int(max_unique_labels_num / 3)
    # batch_loss_dist = []
    # iterating through all classes
    for i in range(0, len(unique_labels)):
        # iterating number of times equal to max image count of all category (doing this step , not to over represent
        # the categories with more images)
        num1 = 0
        num2 = 0
        num3 = 0
        for j in range(0, max_unique_labels_num * 10):
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
            A = conv_model.predict(
                tf.reshape(
                    tf.image.resize(list_images[tmp_a_idx_img], (img_size, img_size)),
                    IMG_SHAPE_NEW,
                )
            )
            P = conv_model.predict(
                tf.reshape(
                    tf.image.resize(list_images[tmp_p_idx_img], (img_size, img_size)),
                    IMG_SHAPE_NEW,
                )
            )
            N = conv_model.predict(
                tf.reshape(
                    tf.image.resize(list_images[tmp_n_idx_img], (img_size, img_size)),
                    IMG_SHAPE_NEW,
                )
            )
            studybatchloss = np.sum(np.square(A - P), axis=1) - np.sum(
                np.square(A - N), axis=1
            )
            # batch_loss_dist.append(studybatchloss[0])
            k = 0
            if studybatchloss[0] <= batch_loss_dist_25 and num1 <= num_limit:
                k = 1
                num1 = num1 + 1
            if (
                studybatchloss[0] > batch_loss_dist_25
                and studybatchloss[0] <= batch_loss_dist_75
                and num2 <= num_limit
            ):
                k = 1
                num2 = num2 + 1
            if studybatchloss[0] > batch_loss_dist_75 and num3 <= num_limit:
                k = 1
                num3 = num3 + 1
            # print(studybatchloss[0],batch_loss_dist_25,batch_loss_dist_75,num1,num2,num3)
            if k == 1:
                list_img_index.append(
                    (list_images[tmp_a_idx_img], list_images[tmp_p_idx_img], [0, 1])
                )
                list_img_label.append([tmp_a_idx, tmp_p_idx])
                list_img_index.append(
                    (list_images[tmp_a_idx_img], list_images[tmp_n_idx_img], [1, 0])
                )
                list_img_label.append([tmp_a_idx, tmp_n_idx])
            if num1 == num_limit and num2 == num_limit and num3 == num_limit:
                break
        # print("Success",i,num1,num2,num3)
        # batch_loss_dist_25 = np.percentile(batch_loss_dist, 25)
        # batch_loss_dist_75 = np.percentile(batch_loss_dist, 75)
        # x = [i for i in batch_loss_dist if i < batch_loss_dist_25]
        # y = [i for i in batch_loss_dist if i > batch_loss_dist_75]
        # print(len(batch_loss_dist),len(x),len(y))
        # print(batch_loss_dist_25, batch_loss_dist_75)
        # exit(0)
    return list_img_index, max_unique_labels_num, list_img_label
