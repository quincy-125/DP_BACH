import openslide
import tensorflow as tf
import os
import argparse
import sys
import numpy as np
from PIL import Image
import io
import matplotlib
from skimage.color import rgb2lab, rgb2hed

matplotlib.use('agg')
import matplotlib.pyplot as plt

from shapely.geometry import Polygon, Point, MultiPoint
from descartes.patch import PolygonPatch


'''function to check if input files exists and valid'''


def input_file_validity(file):
    """Validates the input files"""
    if os.path.exists(file) == False:
        raise argparse.ArgumentTypeError('\nERROR:Path:\n' + file + ':Does not exist')
    if os.path.isfile(file) == False:
        raise argparse.ArgumentTypeError('\nERROR:File expected:\n' + file + ':is not a file')
    if os.access(file, os.R_OK) == False:
        raise argparse.ArgumentTypeError('\nERROR:File:\n' + file + ':no read access ')
    return file


def argument_parse():
    """Parses the command line arguments"""
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-p", "--patch_dir", help="Patch dir", required="True")
    parser.add_argument("-i", "--input_file", help="input file", required="True")
    parser.add_argument("-o", "--tf_output", help="output tf dir", required="True")
    parser.add_argument("-s", "--patch_size", help="Patch_size", required="True")
    parser.add_argument("-l", "--level", help="level", required="True")
    parser.add_argument("-x", "--rgb2hed_thresh", help="level", required="True")
    parser.add_argument("-c", "--mut_type", help="mut type", required="True")
    parser.add_argument("-a", "--threshold_area_percent", help="background Threshold pixel cutoff area percent",
                        required="True")
    parser.add_argument("-t", "--threshold", help="background Threshold pixel cutoff", default="245")
    parser.add_argument("-m", "--threshold_mean", help="background Threshold mean cutoff", default="245")
    parser.add_argument("-d", "--threshold_std", help="background Threshold std cutoff", default="0")
    parser.add_argument("-z", "--patch_byte_cutoff", help="patch_byte_cutoff", default="0")
    return parser


'''creating binary mask to inspect areas with tissue and performance of threshold'''


def create_binary_mask_new(rgb2lab_thresh, svs_file, patch_dir, samp):
    OSobj = openslide.OpenSlide(svs_file)

    divisor = int(OSobj.level_dimensions[0][0] / 500)
    patch_sub_size_x = int(OSobj.level_dimensions[0][0] / divisor)
    patch_sub_size_y = int(OSobj.level_dimensions[0][1] / divisor)
    img = OSobj.get_thumbnail((patch_sub_size_x, patch_sub_size_y))
    toplevel = [patch_sub_size_x, patch_sub_size_y, divisor]
    img = img.convert('RGB')
    np_img = np.array(img)

    img.save(patch_dir + '/' + samp + "_original.png", "png")

    lab_img = rgb2hed(np_img)
    l_img = lab_img[:, :, 2]
    patch_max = round(np.amax(l_img), 2)
    patch_min = round(np.amin(l_img), 2)
    binary_img = l_img > float(rgb2lab_thresh)
    binary_img = binary_img.astype(int)
    np_img[binary_img == 0] = [0, 0, 0]
    np_img[binary_img == 1] = [255, 255, 255]
    im_sub = Image.fromarray(np_img)
    im_sub.save(patch_dir + '/' + samp + "_mask.png", "png")

    idx = np.sum(binary_img)
    mask_area = idx / (binary_img.size)
    print(mask_area)
    idx = np.where(binary_img == 1)
    list_binary_img = []
    for i in range(0, len(idx[0]), 1):
        x = idx[1][i]
        y = idx[0][i]
        list_binary_img.append(str(x) + ' ' + str(y))

    # return np.array(binary_img)
    return list_binary_img, toplevel


'''extracting patch coordinates for requested level based on threshold'''


def calc_patches_cord(list_binary_img, patch_level, svs_file, patch_dir, samp, patch_size, threshold_area_percent,
                      toplevel):
    patch_start_x_list = []
    patch_stop_x_list = []
    patch_start_y_list = []
    patch_stop_y_list = []
    OSobj = openslide.OpenSlide(svs_file)
    minx = 0
    miny = 0
    if patch_level > len(OSobj.level_dimensions) - 1:
        print("not enough levels " + str(patch_level) + " " + str(len(OSobj.level_dimensions) - 1))
        sys.exit(0)
    maxx = OSobj.level_dimensions[patch_level][0]
    maxy = OSobj.level_dimensions[patch_level][1]
    start_x = minx
    total_num_patches = 0
    selected_num_patches = 0

    '''creating sub patches'''
    '''Iterating through x coordinate'''
    while start_x + patch_size < maxx:
        '''Iterating through y coordinate'''
        start_y = miny
        while start_y + patch_size < maxy:
            current_x = int((start_x * OSobj.level_downsamples[patch_level]) / toplevel[2])
            current_y = int((start_y * OSobj.level_downsamples[patch_level]) / toplevel[2])
            tmp_x = start_x + int(patch_size)
            tmp_y = start_y + int(patch_size)
            current_x_stop = int((tmp_x * OSobj.level_downsamples[patch_level]) / toplevel[2])
            current_y_stop = int((tmp_y * OSobj.level_downsamples[patch_level]) / toplevel[2])
            total_num_patches = total_num_patches + 1
            flag_list = [1 for i in range(current_x, current_x_stop + 1) for j in range(current_y, current_y_stop + 1)
                         if str(i) + ' ' + str(j) in list_binary_img]
            if tmp_x <= maxx and tmp_y <= maxy and (len(flag_list) / (
                    (current_y_stop + 1 - current_y) * (current_x_stop + 1 - current_x))) > threshold_area_percent:
                patch_start_x_list.append(start_x)
                patch_start_y_list.append(start_y)
                patch_stop_x_list.append(tmp_x)
                patch_stop_y_list.append(tmp_y)
                selected_num_patches = selected_num_patches + 1

            start_y = tmp_y
        start_x = tmp_x
    print(selected_num_patches, total_num_patches)
    return patch_start_x_list, patch_stop_x_list, patch_start_y_list, patch_stop_y_list


'''creating summary image of toplevel with over lay of selected patches'''


def create_summary_img(patch_start_x_list, patch_stop_x_list, patch_start_y_list, patch_stop_y_list, samp, patch_dir,
                       toplevel, patch_level, svs_file):

    OSobj = openslide.OpenSlide(svs_file)
    poly_included = []
    poly_excluded = []
    name = ""
    for i in range(0, len(patch_stop_x_list), 1):
        x1 = int((patch_start_x_list[i] * OSobj.level_downsamples[patch_level]) / toplevel[2])
        x2 = int((patch_stop_x_list[i] * OSobj.level_downsamples[patch_level]) / toplevel[2])
        y1 = int((patch_start_y_list[i] * OSobj.level_downsamples[patch_level]) / toplevel[2])
        y2 = int((patch_stop_y_list[i] * OSobj.level_downsamples[patch_level]) / toplevel[2])
        poly_included.append(Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]))

    patch_sub_size_x = toplevel[0]
    patch_sub_size_y = toplevel[1]

    img_patch = OSobj.get_thumbnail((patch_sub_size_x, patch_sub_size_y))
    np_img = np.array(img_patch)
    patch_sub_size_y = np_img.shape[0]
    patch_sub_size_x = np_img.shape[1]
    f, ax = plt.subplots(frameon=False)
    f.tight_layout(pad=0, h_pad=0, w_pad=0)
    ax.set_xlim(0, patch_sub_size_x)
    ax.set_ylim(patch_sub_size_y, 0)
    ax.imshow(img_patch)

    for j in range(0, len(poly_included)):
        patch1 = PolygonPatch(poly_included[j], facecolor=[0, 0, 0], edgecolor="green", alpha=0.3, zorder=2)
        ax.add_patch(patch1)
    ax.set_axis_off()
    DPI = f.get_dpi()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    f.set_size_inches(patch_sub_size_x / DPI, patch_sub_size_y / DPI)
    f.savefig(os.path.join(patch_dir, samp + "_mask_patchoverlay.png"), pad_inches='tight')
    return True


'''Quincy code to extract feature vector'''


def patch_feature_extraction_resnet(input_shape=(256, 256, 3)):
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


def patch_feature_extraction(image_string, res50, adaptive_mean_spatial_layer, input_shape=(256, 256, 3)):
    """
    Args:
        image_string:  bytes(PIL_image)
    :return: features:  Feature Vectors, float32
    """

    image_np = np.array(image_string)

    image_batch = np.expand_dims(image_np, axis=0)
    image_patch = tf.keras.applications.resnet50.preprocess_input(image_batch.copy())
    image_patch = tf.image.per_image_standardization(image_patch).numpy()
    ## Return the feature vectors
    predicts = res50.predict(image_patch)
    features = adaptive_mean_spatial_layer(predicts)
    features = tf.io.serialize_tensor(features)
    img_features = features.numpy()

    return img_features


'''TF2 helper functions for TF Records'''


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


'''TF2 helper functions for TF Records'''


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


'''TF2 helper functions for TF Records'''


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


'''extracting patches and creating tfrecords'''


def create_tfrecord(patch_start_x_list, patch_stop_x_list, patch_start_y_list, patch_stop_y_list, samp, patch_dir,
                    patch_level, svs_file, toplevel, tf_output, patch_size, mut_type, threshold_mean, threshold_std,
                    patch_byte_cutoff):

    writer = tf.io.TFRecordWriter(os.path.join(tf_output, samp + '.tfrecords'))

    OSobj = openslide.OpenSlide(svs_file)
    poly_included = []
    res50, adaptive_mean_spatial_layer = patch_feature_extraction_resnet()
    for i in range(0, len(patch_start_x_list), 1):
        x1 = int(patch_start_x_list[i] * OSobj.level_downsamples[patch_level])
        x2 = int(patch_stop_x_list[i] * OSobj.level_downsamples[patch_level])
        y1 = int(patch_start_y_list[i] * OSobj.level_downsamples[patch_level])
        y2 = int(patch_stop_y_list[i] * OSobj.level_downsamples[patch_level])
        img = OSobj.read_region((x1, y1), patch_level, (patch_size, patch_size))
        img = img.convert('RGB')

        '''Change to grey scale'''
        grey_img = img.convert('L')
        '''Convert the image into numpy array'''
        np_grey = np.array(grey_img)
        patch_mean = round(np.mean(np_grey), 2)
        patch_std = round(np.std(np_grey), 2)
        height = patch_size
        width = patch_size
        imgByteArr = io.BytesIO()
        img.save(imgByteArr, format='PNG')
        size_bytes = imgByteArr.tell()
        image_name = samp + "_x_" + str(x1) + "_" + str(x2) + "_y_" + str(y1) + "_" + str(y2) + '_' + str(
            patch_mean) + '_' + str(patch_std) + '_' + str(size_bytes) + ".jpg"
        img.save(patch_dir + '/' + image_name, format='JPEG')
        image_string = open(patch_dir + '/' + image_name, 'rb').read()
        image_feature = patch_feature_extraction(img, res50, adaptive_mean_spatial_layer, (patch_size, patch_size, 3))
        image_format = 'jpeg'
        '''writing tfrecord'''
        feature = {'height': _int64_feature(patch_size),
                   'width': _int64_feature(patch_size),
                   'depth': _int64_feature(3),
                   'label': _int64_feature(mut_type),
                   'image/format': _bytes_feature(image_format.encode('utf8')),
                   'image_name': _bytes_feature(image_name.encode('utf8')),
                   'image/encoded': _bytes_feature(image_string),
                   'image_feature': _bytes_feature(image_feature)}
        os.remove(patch_dir + '/' + image_name)
        Example = tf.train.Example(features=tf.train.Features(feature=feature))
        Serialized = Example.SerializeToString()
        writer.write(Serialized)
        '''preparing the summary thumbnail'''
        x1 = int((patch_start_x_list[i] * OSobj.level_downsamples[patch_level]) / toplevel[2])
        x2 = int((patch_stop_x_list[i] * OSobj.level_downsamples[patch_level]) / toplevel[2])
        y1 = int((patch_start_y_list[i] * OSobj.level_downsamples[patch_level]) / toplevel[2])
        y2 = int((patch_stop_y_list[i] * OSobj.level_downsamples[patch_level]) / toplevel[2])
        poly_included.append(Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]))

    patch_sub_size_x = toplevel[0]
    patch_sub_size_y = toplevel[1]
    img_patch = OSobj.get_thumbnail((patch_sub_size_x, patch_sub_size_y))

    np_img = np.array(img_patch)
    patch_sub_size_y = np_img.shape[0]
    patch_sub_size_x = np_img.shape[1]
    f, ax = plt.subplots(frameon=False)
    f.tight_layout(pad=0, h_pad=0, w_pad=0)
    ax.set_xlim(0, patch_sub_size_x)
    ax.set_ylim(patch_sub_size_y, 0)
    ax.imshow(img_patch)

    for j in range(0, len(poly_included)):
        patch1 = PolygonPatch(poly_included[j], facecolor=[0, 0, 0], edgecolor="green", alpha=0.3, zorder=2)
        ax.add_patch(patch1)
    ax.set_axis_off()
    DPI = f.get_dpi()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    f.set_size_inches(patch_sub_size_x / DPI, patch_sub_size_y / DPI)
    f.savefig(os.path.join(patch_dir, samp + "_mask_patchoverlay_final.png"), pad_inches='tight')


def main():
    abspath = os.path.abspath(__file__)
    words = abspath.split("/")
    '''reading the config filename'''
    parser = argument_parse()
    arg = parser.parse_args()
    '''printing the config param'''
    print("Entered INPUT Filename " + arg.input_file)
    print("Entered Output Patch Directory " + arg.patch_dir)
    print("Entered Output TF Directory " + arg.tf_output)
    print("Entered Patch size " + arg.patch_size)
    print("Entered Level " + arg.level)
    print("Entered background Threshold pixel cutoff " + arg.threshold)
    print("Entered background Threshold pixel cutoff area percent " + arg.threshold_area_percent)
    print("Entered background Threshold mean cutoff " + arg.threshold_mean)
    print("Entered background Threshold std cutoff " + arg.threshold_std)
    print("Entered patch_byte_cutoff " + arg.patch_byte_cutoff)
    print("Entered RGB2labthreshold Threshold std cutoff " + arg.rgb2hed_thresh)
    print("Entered mut type " + arg.mut_type)
    patch_sub_size = int(arg.patch_size)
    rgb2hed_thresh = arg.rgb2hed_thresh
    patch_dir = arg.patch_dir
    tf_output = arg.tf_output

    patch_level = int(arg.level)
    patch_size = int(arg.patch_size)
    threshold = float(arg.threshold)
    mut_type = int(arg.mut_type)
    threshold_area_percent = float(arg.threshold_area_percent)
    svs_file = arg.input_file
    threshold_mean = float(arg.threshold_mean)
    threshold_std = float(arg.threshold_std)
    patch_byte_cutoff = float(arg.patch_byte_cutoff)

    '''Reading TCGA file'''
    samp = os.path.basename(svs_file)

    '''creating binary mask to inspect areas with tissue and performance of threshold'''
    list_binary_img, toplevel = create_binary_mask_new(rgb2hed_thresh, svs_file, patch_dir, samp)
    '''extracting patch coordinates for requested level based on threshold'''
    patch_start_x_list, patch_stop_x_list, patch_start_y_list, patch_stop_y_list = calc_patches_cord(list_binary_img,
                                                                                                     patch_level,
                                                                                                     svs_file,
                                                                                                     patch_dir, samp,
                                                                                                     patch_size,
                                                                                                     threshold_area_percent,
                                                                                                     toplevel)
    '''creating summary image of toplevel with over lay of selected patches'''
    create_summary_img(patch_start_x_list, patch_stop_x_list, patch_start_y_list, patch_stop_y_list, samp, patch_dir,
                       toplevel, patch_level, svs_file)
    '''extracting patches and creating tfrecords'''
    create_tfrecord(patch_start_x_list, patch_stop_x_list, patch_start_y_list, patch_stop_y_list, samp, patch_dir,
                    patch_level, svs_file, toplevel, tf_output, patch_size, mut_type, threshold_mean, threshold_std,
                    patch_byte_cutoff)


if __name__ == "__main__":
    main()
