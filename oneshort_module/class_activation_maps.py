from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
import tensorflow as tf
from tensorflow.keras import models
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import (
    preprocess_input,
    decode_predictions,
)
import cv2

# 1) Compute the model output and last convolutional layer output for the image.
# 2) Find the index of the winning class in the model output.
# 3) Compute the gradient of the winning class with resepct to the last convolutional layer.
# 4) Average this, then weigh it with the last convolutional layer (multiply them).
# 5) Normalize between 0 and 1 for visualization
# 6) Convert to RGB and layer it over the original image.

filepath = "/projects/shart/digital_pathology/results/General-ImageClassifier/tcga_brca1_General-ImageClassifier_2_6_2020/train/ResNet50_SGD_0.001-BinaryCrossentropy/my_model.h5"
new_model = models.load_model(filepath)
IMAGE_SHAPE = (256, 256)

input_folder = "/projects/shart/digital_pathology/scripts/Breast_CHEK2/General-ImageClassifier_pipeline_scripts/sample_patch"
output_folder = "/projects/shart/digital_pathology/scripts/Breast_CHEK2/General-ImageClassifier_pipeline_scripts/sample_patch_cam"
files = os.listdir(input_folder)
layer_name = "conv5_block3_3_conv"

for imgpath in files:
    print(imgpath)
    orig = os.path.join(input_folder, imgpath)
    imgpath = imgpath.replace(".png", "_out.png")
    ###example start
    example = 0
    # example=1
    if example == 1:
        orig = "/projects/shart/digital_pathology/scripts/Breast_CHEK2/General-ImageClassifier_pipeline_scripts/tf-explain/elephant.jpg"
        IMAGE_SHAPE = (299, 299)
        layer_name = "conv2d_93"
        output_folder = "/projects/shart/digital_pathology/scripts/Breast_CHEK2/General-ImageClassifier_pipeline_scripts/tf-explain"
        imgpath = "elephant_out.jpg"
        new_model = InceptionV3(weights="imagenet")
    class_index = 0
    intensity = 0.5
    res = 250
    img = image.load_img(orig, target_size=IMAGE_SHAPE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    if example == 1:
        x = preprocess_input(x)
    else:
        b = K.constant(x)
        b = tf.cast(b, tf.float32)
        b = tf.image.per_image_standardization(b)
        b = tf.image.resize(b, IMAGE_SHAPE)
        b = tf.reshape(b, (IMAGE_SHAPE[0], IMAGE_SHAPE[1], 3))
        x = K.eval(b)
    x = np.expand_dims(x, axis=0)
    preds = new_model.predict(x)
    preds_str = str(round(preds[0][0], 2)) + "_" + str(round(preds[0][1], 2))
    imgpath = imgpath.replace("_out.png", "_" + preds_str + "_out.png")
    if example == 1:
        print(decode_predictions(preds)[0][0][1])  # prints the class of image
    with tf.GradientTape() as tape:
        last_conv_layer = new_model.get_layer(layer_name)
        iterate = tf.keras.models.Model(
            [new_model.inputs], [new_model.output, last_conv_layer.output]
        )
        model_out, last_conv_layer = iterate(x)
        class_out = model_out[:, np.argmax(model_out[0])]
        grads = tape.gradient(class_out, last_conv_layer)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) > 0.0:
        heatmap /= np.max(heatmap)
    heatmap = heatmap.reshape((8, 8))
    img = cv2.imread(orig)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    img = heatmap * intensity + img

    cv2.imwrite(os.path.join(output_folder, imgpath), cv2.resize(img, IMAGE_SHAPE))
    # example=1
    if example == 1:
        sys.exit(0)
