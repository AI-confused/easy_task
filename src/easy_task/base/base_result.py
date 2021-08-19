"""
-*- coding: utf-8 -*-
@author: black_tears
@time: 2021-07-09
@description: base module of task result calculation.
"""


import torch
from sklearn.metrics import *


class BaseResult(object):
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

    @property
    def accuracy(self):
        return round(accuracy_score(self.label, self.pred), 4)

    @property
    def f1_score_0(self):
        return round(f1_score(self.label, self.pred, pos_label=0), 4)

    @property
    def f1_score_1(self):
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

    @property
    def f_05_score(self):
        return round((1 + 0.5 * 0.5) * (self.precision * self.recall) / (0.5 * 0.5 * self.precision + self.recall + 1e-10), 4)

    @property
    def roc_auc(self):
        return round(roc_auc_score(self.label, self.prob), 4)


    def update_batch(self, batch_label: torch.tensor, batch_outputs: torch.tensor):
        """Update batch data in custom task.

        This function need modify in children class.

        @batch_label: batch labels
        @batch_outputs: batch predictions
        """
        pass

