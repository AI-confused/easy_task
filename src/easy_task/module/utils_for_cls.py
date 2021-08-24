"""
-*- coding: utf-8 -*-
@author: black_tears
@time: 2021-07-09
@description: task level function file.
"""


import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from base.base_utils import *
from base.base_result import *

        
class BERTChineseCharacterTokenizer(BertTokenizer):
    """Customized tokenizer for Chinese.
    
    @text: text to tokenize.
    """
    def tokenize(self, text: str) -> list:
        return list(text)

    
class TextDataset(Dataset):
    """Dataset class(custom).
    
    @examples: list of InputFeatures
    """
    def __init__(self, examples: list):
        self.examples = examples
        
    def __len__(self) -> int:
        return len(self.examples)
      
    def __getitem__(self, index: int):
        return self.examples[index]


class ClassificationResult(BaseResult):
    """Store and calculate result class(custom), inherit from BaseResult.

    @task_name: string of task name
    """
    def __init__(self, task_name: str):
        super(ClassificationResult, self).__init__(task_name=task_name)
        
    def get_score(self) -> dict:
        """Calculate task specific score(custom).
        """
        assert len(self.label) == len(self.pred)
        acc = self.accuracy
        prec = self.precision
        rec = self.recall
        f1_score_1 = self.f1_score_1
        f1_score_0 = self.f1_score_0
        macro_f1 = self.macro_f1_score
        micro_f1 = self.micro_f1_score
        f_05 = self.f_05_score
        if(self.prob):
            auc = self.roc_auc
            return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': [f1_score_0, f1_score_1], 'auc': auc, 'f_05_score': f_05}
        else:
            return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': [f1_score_0, f1_score_1], 'micro': micro_f1, 'macro': macro_f1}
        

    def update_batch(self, batch_outputs: torch.Tensor, batch_labels: torch.Tensor, batch_features: list):
        """Update batch result during model eval.

        @batch_outputs: /
        @batch_label: /
        """
        # update batch output
        pred = torch.argmax(batch_outputs, axis=1).cpu().detach().numpy().tolist()
        self.pred += pred
        label = batch_labels.cpu().detach().numpy().tolist()
        self.label += label

        # update batch bad case
        assert len(pred) == len(label) == len(batch_features)
        for _ in range(len(pred)):
            if pred[_] != label[_]:
                self.bad_case['text'].append(batch_features[_].sentence)
                self.bad_case['id'].append(batch_features[_].doc_id)
                self.bad_case['pred'].append(pred[_])
                self.bad_case['label'].append(label[_])