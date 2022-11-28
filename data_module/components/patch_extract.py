# Copyright 2022 Mayo Clinic. All Rights Reserved.
#
# Author: Quincy Gu (M216613)
# Affliation: Division of Computational Pathology and Artificial Intelligence,
# Department of Laboratory Medicine and Pathology, Mayo Clinic College of Medicine and Science
# Email: Gu.Qiangqiang@mayo.edu
# Version: 1.0.1
# Created on: 11/27/2022 11:37 pm CST
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import os
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image

import sys

sys_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(sys_dir))

from components.utils import *

configure_logging(script_name="cross_val_split")


def slide_extract_patches(slide_path, patch_size, patch_path):
    """_summary_

    Args:
        slide_path (_type_): _description_
        patch_size (_type_): _description_
        patch_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    slide = Image.open(slide_path)
    uuid = slide_path.split("/")[-1].split(".")[0]

    slide_width, slide_height = slide.size
    patch_width, patch_height = patch_size

    patches = list()
    for left in range(0, slide_width, patch_width):
        for top in range(0, slide_height, patch_height):
            right, bottom = left + patch_width, top + patch_height
            crop_box_coords = (left, top, right, bottom)

            patch = slide.crop(crop_box_coords)
            patches.append(patch)

            if len(patch_path) > 1:
                slide_patch_path = os.path.join(patch_path, uuid)
                os.makedirs(slide_patch_path, exist_ok=True)
                patch.save("{}/{}_left_{}_top_{}_right_{}_bottom_{}_patch.png".format(slide_patch_path, uuid, left, top, right, bottom))

    expect_num_patches = (slide_width * slide_height) // (patch_width * patch_height)
    assert expect_num_patches == len(patches), logging.debug("We expected to have total number of {} patches been extracted, however, we ended up with only {} patches have been extracted".format(expect_num_patches, len(patches)))

    return patches

def bach_patch_extractions(data_path, kf_csv_path, patch_size, patch_path):
    """_summary_

    Args:
        data_path (_type_): _description_
        kf_csv_path (_type_): _description_
        patch_size (_type_): _description_
        patch_path (_type_): _description_
    """
    import shutil
    
    dest_path = os.path.join(os.path.split(data_path)[0], "tmp")

    patch_size = eval(patch_size)

    logging.info("starting to copy all BACH slides to the tmp folder")
    num_src_files = list()
    for label in tf.io.gfile.listdir(data_path):
        src_path = os.path.join(data_path, label)
        shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
        num_src_files.append(len(tf.io.gfile.listdir(src_path)))
    
    num_copy_files = sum(num_src_files)
    assert num_copy_files == len(tf.io.gfile.listdir(dest_path)), logging.debug("expected to have {} slides copied to the tmp folder {}, however, only {} slides have been copied, there are {} files which have not been copied to the destinated folder".format(num_copy_files, dest_path, len(tf.io.gfile.listdir(dest_path)), num_copy_files - len(tf.io.gfile.listdir(dest_path))))
    logging.info("successfully copied {} slides to the tmp folder {}".format(num_copy_files, dest_path))

    logging.warning("this will do patch extraction for all splits, which will require a large amount of disk space, it is not ideal. recommend provide the path to a specific split csv file rather than the folder with mutiple split csv files.")
    
    for i in tf.io.gfile.listdir(kf_csv_path):
        logging.info("execute patch extraction pipeline for slides in dataset {}".format(i.split(".")[0]))
        slides_df = pd.read_csv(os.path.join(kf_csv_path, i))

        uuids = list(slides_df["UUID"])
        for uuid in uuids:
            logging.info("start patch extraction for slide {}".format(uuid))
            slide_path = os.path.join(dest_path, uuid)

            if len(patch_path) > 1:
                full_patch_path = os.path.join(patch_path, i.split(".")[0])
                os.makedirs(full_patch_path, exist_ok=True)

            slide_extract_patches(
                slide_path=slide_path, 
                patch_size=patch_size, 
                patch_path=full_patch_path
            )
            if len(tf.io.gfile.listdir(kf_csv_path)) == 1:
                logging.debug("remove slide {} from tmp folder".format(slide_path))
                tf.io.gfile.remove(slide_path)

                assert len(tf.io.gfile.listdir(dest_path)) == 0, logging.debug("tmp folder after the completion of patch extraction pipeline supposed to be empty, however, it still has {} files there".format(len(tf.io.gfile.listdir(dest_path))))
    
    tf.io.gfile.rmtree(dest_path)
    logging.info("removed tmp folder {}, the existence of the tmp folder is {}".format(dest_path, tf.io.gfile.exists(dest_path)))

    logging.info("patch extraction task has been completed")