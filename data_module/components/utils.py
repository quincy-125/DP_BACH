# Copyright 2022 Mayo Clinic. All Rights Reserved.
#
# Author: Quincy Gu (M216613)
# Affliation: Division of Computational Pathology and Artificial Intelligence,
# Department of Laboratory Medicine and Pathology, Mayo Clinic College of Medicine and Science
# Email: Gu.Qiangqiang@mayo.edu
# Version: 1.0.1
# Created on: 11/27/2022 9:00 pm CST
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
import sys
import logging

sys_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(sys_dir))


def load_config():
    """_summary_

    Returns:
        _type_: _description_
    """
    import yaml
    from yaml.loader import SafeLoader

    with open("././configs/data_module_config.yaml") as f:
        config = yaml.load(f, Loader=SafeLoader)

    return config


def configure_logging(script_name):
    """_summary_

    Args:
        script_name (_type_): _description_
        stdout_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    config = load_config()
    log_format = "%(asctime)s: %(name)s: %(levelname)s: %(message)s"

    os.makedirs("../logs", exist_ok=True)
    with open(config["stdout_path"], "w") as f:
        pass

    logging.basicConfig(
        level=logging.DEBUG,
        format=log_format,
        filename=config["stdout_path"],
        filemode="w",
    )

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter(log_format))

    logging.getLogger(script_name).addHandler(console)

    return logging.getLogger(script_name)