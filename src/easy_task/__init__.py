"""
-*- coding: utf-8 -*-
@author: black_tears
@time: 2021-08-16
@description: module file.
"""

import os
import sys
cur_dir = os.path.abspath(__file__)
sys.path.append('/'.join(cur_dir.split('/')[:-1]))
from base import base_task, base_result, base_setting, base_utils
from module import utils_for_cls, model_for_cls, task_for_cls, task_for_ner, utils_for_ner, model_for_ner

