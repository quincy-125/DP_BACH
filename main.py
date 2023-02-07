# Copyright 2022 Mayo Clinic. All Rights Reserved.
#
# Author: Quincy Gu (M216613)
# Affliation: Division of Computational Pathology and Artificial Intelligence,
# Department of Laboratory Medicine and Pathology, Mayo Clinic College of Medicine and Science
# Email: Gu.Qiangqiang@mayo.edu
# Version: 1.0.1
# Created on: 11/27/2022 6:35 pm CST
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


import tensorflow as tf
import os
import json

import hydra
from omegaconf import DictConfig, OmegaConf
from training_module.components.model_main import clam_main
from training_module.util import connect_path_arg_gcs


@hydra.main(version_base=None, config_path="configs", config_name="train_config")
def main(cfg: DictConfig) -> None:
    """_summary_

    Args:
        cfg (DictConfig): _description_

    Returns:
        _type_: _description_
    """
    for key, value in cfg.items():
        if value == "None":
            cfg[key] = eval(value)

    connect_path_arg_gcs(cfg)
    clam_main(cfg)


if __name__ == "__main__":
    print(tf.__version__)
    print(
        "Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU"))
    )
    print(tf.config.experimental.list_physical_devices("GPU"))

    main()
