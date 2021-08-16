"""
-*- coding: utf-8 -*-
@author: black_tears
@time: 2021-08-16
@description: run task demo file.
"""


from base.base_utils import *
from base.base_setting import *
from module.task import *
from module.function import *
from module.model import *


if __name__ == '__main__':    
    # init task utils
    task_utils = BaseUtils()

    # init task setting
    task_setting = TaskSetting(task_utils.task_configuration)

    # build custom task
    task = CustomTask(task_setting, load_train=not task_setting.skip_train, load_dev=not task_setting.skip_train)

    # do train
    if not task_setting.skip_train:
        task.output_result['result_type'] = 'Train_mode'
        task.train()
    # do test
    else:
        task.output_result['result_type'] = 'Test_mode'
        task.logger.info('Skip training')
        task.logger.info('Start evaling')

        # you need write testing model name & put it in task_setting.model_dir
        resume_model_name = ''

        # load checkpoint and do eval
        task.resume_eval_at(resume_model_name)