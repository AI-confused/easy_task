"""
-*- coding: utf-8 -*-
@author: LiYunLiang
@time: 2021-07-09
@description: base module of task configuration.
"""


class TaskSetting(object):
    """Base task setting that can be initialized with a dictionary.

    @task_configuration: task configuration parameters.
    """
    def __init__(self, task_configuration: dict):
        self.__update_by_dict(task_configuration)

    def __update_by_dict(self, config_dict):
        for key, val in config_dict.items():
            setattr(self, key, val)

    # def dump_to(self, dir_path, file_name='task_setting.json'):
    #     dump_fp = os.path.join(dir_path, file_name)
    #     default_dump_json(self.__dict__, dump_fp)
