import os

import yaml

from cl_paths import ROOT_PATH


def load_config(config_name):
    config_path = f'{ROOT_PATH}/cl_config'
    with open(os.path.join(config_path, config_name)) as file:
        config = yaml.safe_load(file)

    return config
