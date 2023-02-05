# Copyright 2022 Mayo Clinic. All Rights Reserved.
#
# Author: Quincy Gu (M216613)
# Affliation: Division of Computational Pathology and Artificial Intelligence,
# Department of Laboratory Medicine and Pathology, Mayo Clinic College of Medicine and Science
# Email: Gu.Qiangqiang@mayo.edu
# Version: 1.0.1
# Created on: 11/27/2022 7:51 pm CST
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
import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from components.cross_val_split import *
from components.patch_extract import *

import sys

sys_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(sys_dir))


def data_module_run(cfg):
    """create bash job script file"""
    if cfg.task == "kf_cv_split":
        run_kf_cross_val(
            data_path=cfg.data_path,
            neg_labels=cfg.neg_labels,
            pos_labels=cfg.pos_labels,
            kf_csv_path=cfg.kf_csv_path,
            test_ratio=cfg.test_ratio,
            n_folds=cfg.n_folds,
            kf_shuffle=cfg.kf_shuffle,
            kf_rs=cfg.kf_rs,
        )
    if cfg.task == "patch_extract":
        bach_patch_extractions(
            data_path=cfg.data_path,
            kf_csv_path=cfg.kf_csv_path,
            patch_size=cfg.patch_size,
            patch_path=cfg.patch_path,
        )


@hydra.main(version_base=None, config_path="../configs", config_name="data_config")
def run_main(cfg : DictConfig) -> None:
    """_summary_

    Args:
        cfg (DictConfig): _description_

    Returns:
        _type_: _description_
    """
    for key, value in cfg.items():
        if value == "None":
            cfg[key] = eval(value)
    
    configure_logging(cfg)
    data_module_run(cfg)


if __name__ == "__main__":
    run_main()