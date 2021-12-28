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
from base_module import base_task, base_result, base_setting, base_utils
from task_module import result_for_cls, model_for_cls, task_for_cls, task_for_ner, result_for_ner, model_for_ner
