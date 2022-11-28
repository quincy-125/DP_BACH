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
import subprocess

from components.cross_val_split import *

import sys
sys_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(sys_dir))

configure_logging(script_name="main")


def load_config(config_path):
    """_summary_

    Args:
        config_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    import yaml
    from yaml.loader import SafeLoader

    with open(config_path) as f:
        config = yaml.load(f, Loader=SafeLoader)
        logging.info("\nLoading {}. \nThe customized configuration parameters are in the following \n  {}".format(config_path.split("/")[-1], config))

    return config

def data_module_run(config):
    """create bash job script file"""
    if config["task"] == "kf_cv_split":
        run_kf_cross_val(
            data_path=config["data_path"], 
            neg_labels=config["neg_labels"], 
            pos_labels=config["pos_labels"],
            kf_csv_path=config["kf_csv_path"], 
            test_ratio=config["test_ratio"], 
            n_folds=config["n_folds"], 
            kf_shuffle=config["kf_shuffle"], 
            kf_rs=config["kf_rs"]
        )

def run_main():
    """_summary_
    """
    config = load_config("configs/data_module_config.yaml")
    data_module_run(config=config)


if __name__=="__main__":
    run_main()