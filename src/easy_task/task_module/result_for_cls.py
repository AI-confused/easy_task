"""
-*- coding: utf-8 -*-
@author: black_tears
@time: 2021-07-09
@description: task level function file.
"""

import torch
from base_module.base_utils import *
from base_module.base_result import *


class ClassificationResult(BaseResult):
    """Store and calculate result class(custom), inherit from BaseResult.

    @task_name: string of task name
    """
    def __init__(self, task_name: str):
        super(ClassificationResult, self).__init__(task_name=task_name)
        self.bad_case = {'id': [], 'text': [], 'pred': [], 'label': []}
        self.prediction = {'id': [], 'pred': []}
        
    def get_score(self) -> dict:
        """Calculate task specific score(custom).
        """
        assert len(self.label) == len(self.pred)
        acc = self.accuracy
        prec = self.precision
        rec = self.recall
        f1_score = self.f1_score
        macro_f1 = self.macro_f1_score
        micro_f1 = self.micro_f1_score

        return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1_score, 'micro': micro_f1, 'macro': macro_f1}
        

    def update_batch(self, batch_results: list, **kwargs):
        """Update batch result during model eval.

        @batch_results: [batch_outputs, batch_labels, batch_features]
        """
        # update batch output
        batch_outputs, batch_labels, batch_features = batch_results
        pred = torch.argmax(batch_outputs, axis=-1).cpu().detach().numpy().tolist()
        self.pred += pred
        label = batch_labels.cpu().detach().numpy().tolist()
        self.label += label

        # update prediction
        assert len(pred) == len(label) == len(batch_features)
        for _ in range(len(pred)):
            self.prediction['id'].append(batch_features[_].doc_id)
            self.prediction['pred'].append(pred[_])

        # update batch bad case
        assert len(pred) == len(label) == len(batch_features)
        for _ in range(len(pred)):
            if pred[_] != label[_]:
                self.bad_case['text'].append(batch_features[_].sentence)
                self.bad_case['id'].append(batch_features[_].doc_id)
                self.bad_case['pred'].append(pred[_])
                self.bad_case['label'].append(label[_])