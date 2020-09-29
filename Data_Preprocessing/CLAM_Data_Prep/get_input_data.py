# Import required packages
import tensorflow as tf
import numpy as np

# get patch-level image feature vectors and slide-level labels

"""
We have 2 Input Data Pipelines:
      if patch-level image feature vectors & slide-level labels in tfrecords:
            get required input data from tfrecords
      else:
            get patch-level image arrays and slide-level labels from tfrecords
            feed patch-level image arrays into pre-trained resnet50 model
            get patch-level image feature vectors out
"""


def get_data_from_tf(tf_path):
    feature = {'height': tf.io.FixedLenFeature([], tf.int64),
               'width': tf.io.FixedLenFeature([], tf.int64),
               'depth': tf.io.FixedLenFeature([], tf.int64),
               'label': tf.io.FixedLenFeature([], tf.int64),
               'image/format': tf.io.FixedLenFeature([], tf.string),
               'image_name': tf.io.FixedLenFeature([], tf.string),
               'image/encoded': tf.io.FixedLenFeature([], tf.string),
               'image_feature': tf.io.FixedLenFeature([], tf.string)}

    tfrecord_dataset = tf.data.TFRecordDataset(tf_path)

    def _parse_image_function(key):
        return tf.io.parse_single_example(key, feature)

    CLAM_dataset = tfrecord_dataset.map(_parse_image_function)

    Image_Features = list()

    for tfrecord_value in CLAM_dataset:
        img_features = tf.io.parse_tensor(tfrecord_value['image_feature'], 'float32')
        slide_labels = tfrecord_value['label']
        slide_label = int(slide_labels)
        Image_Features.append(img_features)

    return Image_Features, slide_label


# Input data pipeline#2
def parse_tf(tf_path):
    feature = {'height': tf.io.FixedLenFeature([], tf.int64),
               'width': tf.io.FixedLenFeature([], tf.int64),
               'depth': tf.io.FixedLenFeature([], tf.int64),
               'label': tf.io.FixedLenFeature([], tf.int64),
               'image/format': tf.io.FixedLenFeature([], tf.string),
               'image_name': tf.io.FixedLenFeature([], tf.string),
               'image/encoded': tf.io.FixedLenFeature([], tf.string)}

    tfrecord_dataset = tf.data.TFRecordDataset(tf_path)

    def _parse_image_function(key):
        return tf.io.parse_single_example(key, feature)

    CLAM_dataset = tfrecord_dataset.map(_parse_image_function)

    Img = list()

    for tfrecord_value in CLAM_dataset:
        img = tf.io.decode_image(tfrecord_value['image/encoded'])
        slide_labels = tfrecord_value['label']
        slide_label = int(slide_labels)
        Image_Shape = img.shape

    return Img, Image_Shape, slide_label

def customize_resnet(input_shape=(512, 512, 3)):
    ## Load the ResNet50 model
    resnet50_model = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet',
                                                             input_shape=input_shape)
    resnet50_model.trainable = False  ## Free Training

    ## Create a new Model based on original resnet50 model ended after the 3rd residual block
    layer_name = 'conv4_block1_0_conv'
    res50 = tf.keras.Model(inputs=resnet50_model.input, outputs=resnet50_model.get_layer(layer_name).output)

    ## Add adaptive mean-spatial pooling after the new model
    adaptive_mean_spatial_layer = tf.keras.layers.GlobalAvgPool2D()

    return res50, adaptive_mean_spatial_layer


def get_img_feature(img_array, res50, adaptive_mean_spatial_layer):
    Image_Features = list()
    for i in range(len(img_array)):
        image_np = np.array(img_array[i])
        image_batch = np.expand_dims(image_np, axis=0)
        image_patch = tf.keras.applications.resnet50.preprocess_input(image_batch.copy())
        predicts = res50.predict(image_patch)
        features = adaptive_mean_spatial_layer(predicts)

        img_features = features.numpy()
        img_features = tf.convert_to_tensor(img_features)

        Image_Features.append(img_features)

    return Image_Features
