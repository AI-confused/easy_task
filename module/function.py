"""
-*- coding: utf-8 -*-
@author: black_tears
@time: 2021-07-09
@description: task level function file.
"""


import json
import torch
import random
import tqdm
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


def read_examples(input_file: str, percent: float=1.0) -> list:
    """Read data from a data file and generate list of InputExamples(custom).

    @input_file: data input file abs dir
    @percent: percent of reading samples
    """
    examples=[]
    cnt = 10000
    with open(input_file) as fin:
        data = json.load(fin)['data']
        if percent != 1.0:
            data = random.sample(data, int(len(data)*percent))
        for line in data:
            text = line['text']
            label = line['label']
            doc_id = cnt
            cnt += 1
            examples.append(InputExample(
                doc_id=doc_id, 
                text=text,
                label=label))
    return examples



def convert_examples_to_features(examples: list, tokenizer: BertTokenizer, max_seq_len: int, **kwargs) -> list:
    """Process the InputExamples into InputFeatures that can be fed into the model(custom).

    @examples: list of InputExamples
    @tokenizer: class BertTokenizer or its inherited classes
    @max_seq_len: max length of tokenized text
    """
    results = []
    for _ in tqdm.tqdm(range(len(examples)), total=len(examples)):
        example = examples[_]

        # tokenize
        sentence_token = tokenizer.tokenize(example.text)[:max_seq_len-2]
        sentence_len = len(sentence_token)
        input_token = ['[CLS]'] + sentence_token + ['[SEP]']
        segment_id = [0] * len(input_token)
        input_id = tokenizer.convert_tokens_to_ids(input_token)
        input_mask = [1] * len(input_id)

        # padding
        padding_length = max_seq_len - len(input_id)
        input_id += ([0] * padding_length)
        input_mask += ([0] * padding_length)
        segment_id += ([0] * padding_length)

        results.append(
            InputFeature(
                doc_id=example.doc_id,
                sentence=example.text,
                input_tokens=input_token,
                input_ids=input_id,
                input_masks=input_mask,
                segment_ids=segment_id,
                sentence_len=sentence_len,
                label=example.label
            )
        )
    return results

    
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


def train_collate_fn(examples: list) -> list:
    """Convert batch training examples into batch tensor(custom).

    @examples(InputFeature): /
    """
    input_ids = torch.stack([torch.tensor(example.input_ids, dtype=torch.long) for example in examples],0)
    input_masks = torch.stack([torch.tensor(example.input_masks, dtype=torch.long) for example in examples],0)
    segment_ids = torch.stack([torch.tensor(example.segment_ids, dtype=torch.long) for example in examples],0)
    labels = torch.stack([torch.tensor(example.label, dtype=torch.long) for example in examples],0)

    return (input_ids, input_masks, segment_ids, labels)


def eval_collate_fn(examples: list) -> list:
    """Convert batch eval examples into batch tensor(custom).

    @examples(InputFeature): /
    """
    input_ids = torch.stack([torch.tensor(example.input_ids, dtype=torch.long) for example in examples],0)
    input_masks = torch.stack([torch.tensor(example.input_masks, dtype=torch.long) for example in examples],0)
    segment_ids = torch.stack([torch.tensor(example.segment_ids, dtype=torch.long) for example in examples],0)
    labels = torch.stack([torch.tensor(example.label, dtype=torch.long) for example in examples],0)

    return (input_ids, input_masks, segment_ids, labels)



class Result(BaseResult):
    """Store and calculate result class(custom), inherit from BaseResult.

    @task_name: string of task name
    """
    def __init__(self, task_name: str):
        super(Result, self).__init__(task_name=task_name)
        
    def get_score(self) -> dict:
        """Calculate task specific score(custom).
        """
        assert len(self.label) == len(self.pred)
        acc = self.accuracy
        prec = self.precision
        rec = self.recall
        f1 = self.micro_f1_score
        f_05 = self.f_05_score
        if(self.prob):
            auc = self.roc_auc
            return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1, 'auc': auc, 'f_05_score': f_05}
        else:
            return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1, 'f_05_score': f_05}
        

    def update_batch(self, batch_outputs: torch.Tensor, batch_labels: torch.Tensor):
        """Update batch result during model eval.

        @batch_outputs: /
        @batch_label: /
        """
        self.pred += torch.argmax(batch_outputs, axis=1).cpu().detach().numpy().tolist()
        self.label += batch_labels.cpu().detach().numpy().tolist()
        
