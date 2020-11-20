import json
import yaml
# from ruamel.yaml import YAML


def load_json(filepath):
    with open(filepath, 'r') as f:
        dict_obj = json.load(f)
    return dict_obj


def load_yaml(filepath):
    with open(filepath, 'r') as f:
        dict_obj = yaml.load(f)
    return dict_obj