# -*- coding: utf-8 -*-
# AUTHOR: Li Yun Liang
# DATE: 21-7-9


import argparse
import yaml
import os
import sys
import torch.distributed as dist
sys.path.append('/home/work/gitlab/thesis_revises/')
from base.base_utils import *
from base.base_setting import *
from module.task import *
from module.function import *
from module.model import *


if __name__ == '__main__':
    set_basic_log_config()
    
    # load config yaml file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True, type=str, help='Please input task specific config yaml file path!')
    args = parser.parse_args()
    with open(args.config_path, 'r', encoding='utf-8') as f:
        task_configuration = yaml.load(f.read(), Loader=yaml.FullLoader)

    # init task setting
    task_setting = TaskSetting(task_configuration)

    # create task output path
    task_dir = os.path.join(task_setting.exp_dir, task_setting.task_name)
    if not os.path.exists(task_dir):
        os.makedirs(task_dir, exist_ok=True)
    task_setting.task_dir = task_dir
    task_setting.model_dir = os.path.join(task_dir, "Model")
    if not os.path.exists(task_setting.model_dir):
        os.makedirs(task_setting.model_dir, exist_ok=True)
    task_setting.output_dir = os.path.join(task_dir, "Output")
    if not os.path.exists(task_setting.output_dir):
        os.makedirs(task_setting.output_dir, exist_ok=True)

    # build custom task
    task = CustomTask(task_setting, load_train=not task_setting.skip_train, load_dev=not task_setting.skip_train)

    # do train
    if not task_setting.skip_train:
        task.train()
    # do eval
    else:
        task.logging('Skip training')
        task.logging('Start evaling')
        # resume_model_dir = os.path.join(in_argv.exp_dir, '{}/Model/chinese_wwm_ext_pytorch_1e-05_40_None_test.pt'.format(in_argv.task_name))
        resume_model_dir = os.path.join(task_setting.exp_dir, '{}/Model/PretrainedBert_2e-06_16_None.pt'.format(in_argv.task_name))
        # load checkpoint and do eval
        task.resume_save_eval_at(resume_model_dir)
        