# -*- coding: utf-8 -*-
# AUTHOR: Li Yun Liang
# DATE: 21-7-9

import torch
from sklearn.metrics import *


class BaseResult(object):
    """
    存储并且计算最终答案分数的基础类
    """
    def __init__(self, task_name):
        self.task_name = task_name
        self.label = []
        self.pred = []
        self.prob = []

    @property
    def accuracy(self):
        return round(accuracy_score(self.label, self.pred), 4)

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


    def update_batch(self, batch_label, batch_outputs):
        """
        update batch data in classification task.
        Args:
            batch_label():
            batch_outputs():
        """
        pass

