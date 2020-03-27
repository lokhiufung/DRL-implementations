import math
import logging
from logging import FileHandler, StreamHandler, Formatter
import os


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


def compare_weights(model1, model2):
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        if param1.data.ne(param2.data).sum() > 0:
            return False
    return True


def sigmoid_clipping(x, beta=1.0):
    return 1 / (1 + math.exp(-beta * x))