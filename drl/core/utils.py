import json
import yaml
# from ruamel.yaml import YAML

from hydra.utils import instantiate
from importlib import import_module


def load_json(filepath):
    with open(filepath, 'r') as f:
        dict_obj = json.load(f)
    return dict_obj


def load_yaml(filepath):
    with open(filepath, 'r') as f:
        dict_obj = yaml.load(f)
    return dict_obj


def name_import(name):
    module_class = import_module(name)
    return module_class


def create_network(cfg):
    pass