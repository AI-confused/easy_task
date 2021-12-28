"""
-*- coding: utf-8 -*-
@author: black_tears
@time: 2021-07-09
@description: base module of task result calculation.
"""


import abc
import torch
from sklearn.metrics import *


class BaseResult(metaclass=abc.ABCMeta):
    """Base class of store & calculate task result.

    The labels and predicted values of each batch are updated one at a time, 
    and the result of the whole dataset is finally calculated.

    @task_name: name of custom task.
    """
    def __init__(self, task_name: str):
        self.task_name = task_name
        self.label = []
        self.pred = []
        self.prob = []
        self.bad_case = {}

    @property
    def accuracy(self):
        return round(accuracy_score(self.label, self.pred), 4)

    @property
    def f1_score(self):
        return round(f1_score(self.label, self.pred, pos_label=1), 4)

    @property
    def micro_f1_score(self):
        return round(f1_score(self.label, self.pred, average='micro'), 4)

    @property
    def macro_f1_score(self):
        return round(f1_score(self.label, self.pred, average='macro'), 4)

    @property
    def precision(self):
        return round(precision_score(self.label, self.pred), 4)

    @property
    def recall(self):
        return round(recall_score(self.label, self.pred), 4)

    @abc.abstractclassmethod
    def update_batch(self, **kwargs):
        """Update batch data in custom task.

        This function must be written by inherit class.
        """
        pass


