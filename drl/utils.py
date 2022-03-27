import json
import yaml
import json
import logging
from logging import FileHandler, StreamHandler, Formatter
import os

from importlib import import_module


LOG_LV_MAPPER = {
    'info': logging.INFO,
    'debug': logging.DEBUG,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}


def get_logger(name, fh_lv='debug', ch_lv='error', logger_lv='debug', logdir='./log'):
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LV_MAPPER[logger_lv])  # set log level of logger

    if not os.path.exists(logdir):
        print('{logdir} does not exist. Make {logdir}'.format(logdir=logdir))
        os.mkdir(logdir)

    fh = FileHandler(logdir + '/' + name + '.log')  
    ch = StreamHandler()
    formatter = Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    fh.setLevel(LOG_LV_MAPPER[fh_lv])  # set log level of filehandler
    ch.setLevel(LOG_LV_MAPPER[ch_lv])
    logger.addHandler(fh)  # set log level of streamhandler
    logger.addHandler(ch)
    return logger


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
