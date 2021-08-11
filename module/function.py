# -*- coding: utf-8 -*-
# AUTHOR: Li Yun Liang
# DATE: 21-7-9


from base.base_result import BaseResult
import json
from pathlib import Path
import time
import torch
import torch.nn as nn
import numpy as np
import random
import pandas as pd
import re
import itertools
from collections import Counter
from typing import Callable, Dict, List, Generator, Tuple
from multiprocessing import Pool
import os
import operator
import tqdm
import logging
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler,TensorDataset, Dataset
from base.base_utils import *
from base.base_result import *
from collections import defaultdict
import copy

        
def set_basic_log_config():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)


def read_examples(input_file, percent=1.0):
    """
    从数据集读取数据，生成文档样本(custom)
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



def convert_examples_to_features(examples, tokenizer, max_seq_len, **kwargs):
    """
    把原始样本处理为可以输入模型的样本(custom)
    """
    results = []
    for _ in tqdm.tqdm(range(len(examples)), total=len(examples)):
        example = examples[_]
        if len(example.text) > max_seq_len - 2:#丢弃文本长度超过78的样本
            continue
        # max_doc_length = max_seq_length - 2
        # features = []
        # doc = example.text
        
        # if is_training==1: # train     
        #     sentence_token = tokenizer.tokenize(example.text)
        #     sentence_len = len(sentence_token)
        #     input_token = ['[CLS]'] + sentence_token + ['[SEP]']
        #     segment_id = [0] * len(input_token)
        #     input_id = tokenizer.convert_tokens_to_ids(input_token)
        #     input_mask = [1] * len(input_id)
        #     #padding
        #     padding_length = max_seq_length - len(input_id)
        #     input_id += ([0] * padding_length)
        #     input_mask += ([0] * padding_length)
        #     segment_id += ([0] * padding_length)
        # elif is_training==2: # eval
        sentence_token = tokenizer.tokenize(example.text)
        sentence_len = len(sentence_token)
        input_token = ['[CLS]'] + sentence_token + ['[SEP]']
        segment_id = [0] * len(input_token)
        input_id = tokenizer.convert_tokens_to_ids(input_token)
        input_mask = [1] * len(input_id)
        #padding
        padding_length = max_seq_len - len(input_id)
        input_id += ([0] * padding_length)
        input_mask += ([0] * padding_length)
        segment_id += ([0] * padding_length)

        results.append(
            InputFeature(
                doc_id=example.doc_id,
                sentence=example.text,
                # arguments=example.arguments,
                input_tokens=input_token,
                input_ids=input_id,
                input_masks=input_mask,
                segment_ids=segment_id,
                sentence_len=sentence_len,
                label=example.label
                # event_type=example.event_type,
            )
        )
        # assert len(features) == len(sentences)
        # results.append(features)
    return results

def default_dump_json(path, content):
    with open(path, 'w') as f:
        f.write(json.dumps(content, ensure_ascii=False))

    
class TextDataset(Dataset):
    def __init__(self, examples: List[InputFeature]):
        self.examples = examples
        
    def __len__(self) -> int:
        return len(self.examples)
      
    def __getitem__(self, index):
        return self.examples[index]


def train_collate_fn(examples):
    """
    convert batch training examples into batch tensor(custom).
    Args:
        examples(InputFeature): 
    """
    input_ids = torch.stack([torch.tensor(example.input_ids, dtype=torch.long) for example in examples],0)
    input_masks = torch.stack([torch.tensor(example.input_masks, dtype=torch.long) for example in examples],0)
    segment_ids = torch.stack([torch.tensor(example.segment_ids, dtype=torch.long) for example in examples],0)
    labels = torch.stack([torch.tensor(example.label, dtype=torch.long) for example in examples],0)

    return [input_ids, input_masks, segment_ids, labels]


def eval_collate_fn(examples):
    """
    convert batch eval examples into batch tensor(custom).
    Args:
        examples(InputFeature): 
    """
    input_ids = torch.stack([torch.tensor(example.input_ids, dtype=torch.long) for example in examples],0)
    input_masks = torch.stack([torch.tensor(example.input_masks, dtype=torch.long) for example in examples],0)
    segment_ids = torch.stack([torch.tensor(example.segment_ids, dtype=torch.long) for example in examples],0)
    labels = torch.stack([torch.tensor(example.label, dtype=torch.long) for example in examples],0)

    return [input_ids, input_masks, segment_ids, labels]



class Result(BaseResult):
    """
    存储并且计算最终答案分数的类,可以自定义
    """
    def __init__(self, task_name):
        super(Result, self).__init__(task_name=task_name)
        
    def get_score(self):
        """
        calculate task specific score(custom)
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
        

    def update_batch(self, batch_outputs, batch_labels):
        """
        update batch result during model eval.
        Args:
            batch_outputs(torch.Tensor): /
            batch_label(torch.Tensor): /
        """
        self.pred += torch.argmax(batch_outputs, axis=1).cpu().detach().numpy().tolist()
        self.label += batch_labels.cpu().detach().numpy().tolist()
        
