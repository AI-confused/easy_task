"""
-*- coding: utf-8 -*-
@author: black_tears
@time: 2021-07-09
@description: base module of task utils.
"""


import json
from torch.utils.data import Dataset
from transformers import BertTokenizer
import pickle
import yaml


class BaseUtils(object):
    """Base class for task utils.
    """
    def __init__(self, task_config_path):
        with open(task_config_path, 'r', encoding='utf-8') as f:
            self.task_configuration = yaml.load(f.read(), Loader=yaml.FullLoader)


    @classmethod
    def default_load_json(json_file_path: str, encoding: str='utf-8', **kwargs):
        """Load json file.

        @json_file_path: json file abs path
        """
        with open(json_file_path, 'r', encoding=encoding) as fin:
            tmp_json = json.load(fin, **kwargs)
        return tmp_json

    @classmethod
    def default_dump_json(obj, json_file_path: str, encoding: str='utf-8', ensure_ascii: bool=False, indent: int=2, **kwargs):
        """Dump json contents to file.

        @obj: json content
        @json_file_path: json file abs path
        @ensure_ascii: ascii code or not in json file
        @indent: /
        """
        with open(json_file_path, 'w', encoding=encoding) as fout:
            json.dump(obj, fout,
                    ensure_ascii=ensure_ascii,
                    indent=indent,
                    **kwargs)
            fout.write('\n')

    @classmethod
    def write_lines(obj, file_path: str, content: list, write_type: str='a', encoding: str='utf-8'):
        """Add line to file.
        
        @file_path: /
        @content: list of line
        """
        with open(file_path, write_type, encoding=encoding) as fout:
            for line in content:
                fout.write(line + '\n')

    @classmethod
    def default_load_pkl(pkl_file_path: str, **kwargs):
        """Load json file.

        @pkl_file_path: pickle file abs path
        """
        with open(pkl_file_path, 'rb') as fin:
            obj = pickle.load(fin, **kwargs)

        return obj

    @classmethod
    def default_dump_pkl(obj, pkl_file_path: str, **kwargs):
        """Dump pickle file to path.

        @pkl_file_path: pickle file abs path
        """
        with open(pkl_file_path, 'wb') as fout:
            pickle.dump(obj, fout, **kwargs)


class InputExample(object):
    """A raw input example.

    @kwargs: custom attributes of class
    """
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

class InputFeature(object):
    """A feature example to input model.

    @kwargs: custom attributes of class
    """
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


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


class BERTChineseCharacterTokenizer(BertTokenizer):
    """Customized tokenizer for Chinese.
    
    @text: text to tokenize.
    """
    def tokenize(self, text: str) -> list:
        return list(text)
