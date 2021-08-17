"""
-*- coding: utf-8 -*-
@author: black_tears
@time: 2021-07-09
@description: base module of task configuration.
"""

import os

class TaskSetting(object):
    """Base task setting that can be initialized with a dictionary.

    @task_configuration: task configuration parameters.
    """
    def __init__(self, task_configuration: dict):
        self.__update_by_dict(task_configuration)
        self.__create_task_output_dir()

    def __update_by_dict(self, config_dict: dict):
        """Update class attributes by dict.
        
        @config_dict: configuration dictionary
        """
        for key, val in config_dict.items():
            setattr(self, key, val)


    def __create_task_output_dir(self):
        """Create task output path.
        """
        # create task dir
        self.task_dir = os.path.join(self.exp_dir, self.task_name)
        if not os.path.exists(self.task_dir):
            os.makedirs(self.task_dir, exist_ok=True)

        # create model saving dir
        self.model_dir = os.path.join(self.task_dir, "Model")
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir, exist_ok=True)
            
        # create model output dir
        self.output_dir = os.path.join(self.task_dir, "Output")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
