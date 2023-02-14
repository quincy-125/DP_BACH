import tensorflow as tf


class DataRunner:
    def __init__(self, image_file_list):
        super().__init__()
        self.image_file_list = image_file_list

    @staticmethod
    def format_example(image_name=None, img_size=256, train=True):
        """
        Apply any image preprocessing here
        :param image_name: the specific filename of the image
        :param img_size: size that images should be reshaped to
        :param train: whether this is for training or not

        :return: image
        """
        image = tf.io.read_file(image_name)
        image = tf.io.decode_jpeg(image)
        image = tf.cast(image, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.image.resize(image, (img_size, img_size))

        if train is True:
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.0, upper=0.1)
            image = tf.image.random_flip_up_down(image)
            image = tf.image.random_hue(image, max_delta=0.2)

        image = tf.reshape(image, (img_size, img_size, 3))

        return image.numpy()

    def get_distributed_datasets(self):
        for i in self.image_file_list:
            # v1=[]
            # if i[2] == "0":
            # ky = {"anchor_img": i[0], "other_img": i[1]}
            # vl = [0,1]
            # if i[2] == "1":
            # ky = {"anchor_img": i[0], "other_img": i[1]}
            # vl = [1, 0]
            # yield {"anchor_img": i[0], "pos_img": i[1], "neg_img": i[2]}, [1, 1, 0]
            yield {"anchor_img": i[0], "other_img": i[1]}, i[2]
