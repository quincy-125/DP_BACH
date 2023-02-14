# !/usr/local/biotools/python/3.4.3/bin/python3
__author__ = "Naresh Prodduturi"
__email__ = "prodduturi.naresh@mayo.edu"
__status__ = "Dev"

import openslide
import tensorflow as tf
import os
import argparse
import sys
import pwd
import time
import subprocess
import re
import shutil
import glob
import numpy as np
from PIL import Image, ImageDraw
import tempfile
import math
import io
import re
import matplotlib

# from skimage.filters import threshold_otsu
# from skimage.color import rgb2lab,rgb2hed
matplotlib.use("agg")
import matplotlib.pyplot as plt

# from dataset_utils import *
from shapely.geometry import Polygon, Point, MultiPoint
from shapely.geometry import geo
from descartes.patch import PolygonPatch


"""function to check if input files exists and valid"""


def concat_images(imga, imgb):
    """
    Combines two color image ndarrays side-by-side.
    """

    ha, wa = imga.shape[:2]
    hb, wb = imgb.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa + wb
    new_img = np.zeros(shape=(max_height, total_width, 3))
    new_img[:ha, :wa] = imga
    new_img[:hb, wa : wa + wb] = imgb
    return new_img


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
    parser.add_argument("-p", "--patch_dir", help="Patch dir", required="True")
    parser.add_argument("-i", "--input_file", help="input file", required="True")
    parser.add_argument("-l", "--level", help="level", required="True")
    parser.add_argument(
        "-m", "--input_label_file", help="input input_label_file", required="True"
    )
    return parser


def create_binary_mask_new(input_label_file, svs_file, patch_dir, patch_level):
    # print(input_label_file,svs_file,patch_dir,patch_level)
    fn = os.path.basename(svs_file)
    OSobj = openslide.OpenSlide(svs_file)
    divisor = int(OSobj.level_dimensions[0][0] / 500)
    patch_sub_size_x = int(OSobj.level_dimensions[0][0] / divisor)
    patch_sub_size_y = int(OSobj.level_dimensions[0][1] / divisor)
    img = OSobj.get_thumbnail((patch_sub_size_x, patch_sub_size_y))
    img = img.convert("RGB")
    img.save(os.path.join(patch_dir, fn + "_1.png"), "png")
    np_img = np.array(img)

    # poly_included_0 = []
    # poly_included_1 = []
    # fobj=open(input_label_file)
    # for i in fobj:
    # i = i.strip()
    # arr1 = i.split(" ")
    # arr = arr1[1].split("_")
    # x1 = int(arr[len(arr)-8])/divisor
    # x2 = int(arr[len(arr)-7])/divisor
    # y1 = int(arr[len(arr)-5])/divisor
    # y2 = int(arr[len(arr)-4])/divisor
    # if float(arr1[5]) > 0.5:
    # poly_included_1.append(Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]))
    # else:
    # poly_included_0.append(Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]))

    patch_sub_size_y = np_img.shape[0]
    patch_sub_size_x = np_img.shape[1]
    f, ax = plt.subplots(frameon=False)
    f.tight_layout(pad=0, h_pad=0, w_pad=0)
    ax.set_xlim(0, patch_sub_size_x)
    ax.set_ylim(patch_sub_size_y, 0)
    ax.imshow(img)

    poly_included_0 = []
    poly_included_1 = []
    fobj = open(input_label_file)
    for i in fobj:
        i = i.strip()
        arr1 = i.split(" ")
        arr = arr1[1].split("_")
        x1 = int(arr[len(arr) - 8]) / divisor
        x2 = int(arr[len(arr) - 7]) / divisor
        y1 = int(arr[len(arr) - 5]) / divisor
        y2 = int(arr[len(arr) - 4]) / divisor
        if float(arr1[5]) > 0.4:
            tmp = round(float(arr1[5]), 2)
            patch1 = PolygonPatch(
                Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]),
                facecolor=[0, 0, 0],
                edgecolor="red",
                alpha=tmp,
                zorder=2,
            )
            ax.add_patch(patch1)
        else:
            tmp = round(float(arr1[5]), 2)
            patch1 = PolygonPatch(
                Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]),
                facecolor=[0, 0, 0],
                edgecolor="green",
                alpha=tmp,
                zorder=2,
            )
            ax.add_patch(patch1)
    fobj.close()
    # for j in range(0, len(poly_included_0)):
    # patch1 = PolygonPatch(poly_included_0[j], facecolor=[0, 0, 0], edgecolor="green", alpha=0.1, zorder=2)
    # ax.add_patch(patch1)
    # for j in range(0, len(poly_included_1)):
    # patch1 = PolygonPatch(poly_included_1[j], facecolor=[0, 0, 0], edgecolor="red", alpha=0.30, zorder=2)
    # ax.add_patch(patch1)
    ax.set_axis_off()
    DPI = f.get_dpi()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    f.set_size_inches(patch_sub_size_x / DPI, patch_sub_size_y / DPI)
    f.savefig(os.path.join(patch_dir, fn + "_2.png"), pad_inches="tight")
    # images = [os.path.join(patch_dir , fn + "_1.png"),os.path.join(patch_dir , fn + "_2.png")]
    # imga=Image.open(os.path.join(patch_dir , fn + "_1.png"))
    # imgb=Image.open(os.path.join(patch_dir , fn + "_2.png"))
    # imga = imga.convert('RGB')
    # imgb = imgb.convert('RGB')
    # np_imga = np.array(imga)
    # np_imgb = np.array(imgb)
    # output = concat_images(np_imga,np_imgb)
    # img = Image.fromarray(output, 'RGB')
    # img.save(os.path.join(patch_dir , fn + ".png"))
    images = [
        Image.open(x)
        for x in [
            os.path.join(patch_dir, fn + "_1.png"),
            os.path.join(patch_dir, fn + "_2.png"),
        ]
    ]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new("RGB", (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    new_im.save(os.path.join(patch_dir, fn + ".png"), "png")
    os.remove(os.path.join(patch_dir, fn + "_1.png"))
    os.remove(os.path.join(patch_dir, fn + "_2.png"))


def main():
    abspath = os.path.abspath(__file__)
    words = abspath.split("/")
    """reading the config filename"""
    parser = argument_parse()
    arg = parser.parse_args()
    """printing the config param"""
    print("Entered INPUT Filename " + arg.input_file)
    print("Entered INPUT Label Filename " + arg.input_label_file)
    print("Entered Output Patch Directory " + arg.patch_dir)
    print("Entered Level " + arg.level)
    patch_dir = arg.patch_dir
    patch_level = int(arg.level)
    svs_file = arg.input_file
    """Reading TCGA file"""
    samp = os.path.basename(svs_file)

    """creating binary mask to inspect areas with tissue and performance of threshold"""
    create_binary_mask_new(arg.input_label_file, svs_file, patch_dir, patch_level)


if __name__ == "__main__":
    main()
