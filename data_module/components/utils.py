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


def configure_logging(cfg):
    """_summary_

    Args:
        cfg (_type_): _description_

    Returns:
        _type_: _description_
    """
    log_format = "%(asctime)s: %(name)s: %(levelname)s: %(message)s"

    os.makedirs("../logs", exist_ok=True)
    with open(cfg.stdout_path, "w") as f:
        pass

    logging.basicConfig(
        level=logging.DEBUG,
        format=log_format,
        filename=cfg.stdout_path,
        filemode="w",
    )

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter(log_format))

    return logging.getLogger()
