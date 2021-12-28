"""
-*- coding: utf-8 -*-
@author: black_tears
@time: 2021-09-27
@description: run sequence tagging task demo file.
"""

import os
import sys
######## if you have install easy-task package, folowing two lines can be disable ##########
cur_dir = os.path.abspath(__file__)
sys.path.append(os.path.join('/'.join(cur_dir.split('/')[:-2]), 'src/'))
############################################################################################
from easy_task.base_module import TaskSetting, BaseUtils
from easy_task.task_module import SequenceTaggingTask


if __name__ == '__main__':    
    # init task utils
    task_utils = BaseUtils(task_config_path=os.path.join(os.getcwd(), 'task_config/ner_config.yml'))

    # init task setting
    task_setting = TaskSetting(task_utils.task_configuration)

    # build custom task
    task = SequenceTaggingTask(task_setting, load_train=not task_setting.skip_train, load_dev=not task_setting.skip_train)

    # do train
    if not task_setting.skip_train:
        task.output_result['result_type'] = 'Train_mode'
        task.train()
    # do test
    else:
        task.output_result['result_type'] = 'Test_mode'
        task.logger.info('Skip training')
        task.logger.info('Start evaling')

        # load checkpoint and do eval
        task.resume_test_at(task.setting.resume_model_path)