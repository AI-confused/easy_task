# -*- coding: utf-8 -*-
# AUTHOR: Li Yun Liang
# DATE: 21-7-9


import logging
import random
import os
import json
import sys
import numpy as np
from datetime import datetime
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.nn.parallel as para
# from pytorch_pretrained_bert.optimization import BertAdam
from tqdm import trange, tqdm
# from tensorboardX import SummaryWriter

from .base_utils import default_dump_pkl, default_dump_json


class TaskSetting(object):
    # __slots__ = ('model_dir', 'output_dir', 'task_dir')
    """
    Base task setting that can be initialized with a dictionary
    Args:
        key_attrs (list): key attributes.
        attr_default_pairs (list): default attribute pairs.
        kwargs (dict): /
    """
    def __init__(self, task_configuration: dict):
        self.__update_by_dict(task_configuration)

    def __update_by_dict(self, config_dict):
        for key, val in config_dict.items():
            setattr(self, key, val)

    # def dump_to(self, dir_path, file_name='task_setting.json'):
    #     dump_fp = os.path.join(dir_path, file_name)
    #     default_dump_json(self.__dict__, dump_fp)
