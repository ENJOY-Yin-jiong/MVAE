from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import yaml
from easydict import EasyDict as edict

config = edict()

# running set
config.WORKERS = 16
config.LOG_DIR = ''
config.MODEL_DIR = ''

# model
config.MODEL = edict()
config.MODEL.NAME = ''
config.MODEL.CHECKPOINT = ''  # The checkpoint for the best performance
config.MODEL.PARAMS = None

# Dataset
config.DATASET = edict()



# training
config.TRAIN = edict()
config.TRAIN.LR = 0.001
config.TRAIN.WEIGHT_DECAY = 0.0001
config.TRAIN.FACTOR = 0.8
config.TRAIN.PATIENCE = 5
config.TRAIN.GAMMA = 0.5
config.TRAIN.MILE_STONE = [10, 15]
config.TRAIN.MAX_EPOCH = 20
config.TRAIN.BATCH_SIZE = 4
config.TRAIN.PER_NEGATIVE_PAIRS_INBATCH = 3
config.TRAIN.SHUFFLE = True
config.TRAIN.CONTINUE = False

# loss
config.LOSS = edict()

# test
config.TEST = edict()


def _update_dict(cfg, value):
    for k, v in value.items():
        if k in cfg:
            if k == 'PARAMS':
                cfg[k] = v
            elif isinstance(v, dict):
                _update_dict(cfg[k], v)
            else:
                cfg[k] = v
        else:
            raise ValueError("{} not exist in config.py".format(k))


def update_config(config_file):
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    _update_dict(config[k], v)
                else:
                    config[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))