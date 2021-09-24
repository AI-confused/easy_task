"""
-*- coding: utf-8 -*-
@author: black_tears
@time: 2021-09-23
@description: task level function file.
"""

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


class SequenceTaggingResult(BaseResult):
    """Store and calculate result class(custom), inherit from BaseResult.

    @task_name: string of task name
    """
    def __init__(self, task_name: str):
        super(SequenceTaggingResult, self).__init__(task_name=task_name)
        
    def get_score(self) -> dict:
        """Calculate task specific score(custom).
        """
        pass
        

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