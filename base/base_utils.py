# -*- coding: utf-8 -*-
# AUTHOR: Li Yun Liang
# DATE: 21-7-9

import json
import logging
import pickle
import random
import tqdm
from transformers import BertTokenizer


logger = logging.getLogger(__name__)

EPS = 1e-10


def default_load_json(json_file_path, encoding='utf-8', **kwargs):
    with open(json_file_path, 'r', encoding=encoding) as fin:
        tmp_json = json.load(fin, **kwargs)
    return tmp_json


def default_dump_json(obj, json_file_path, encoding='utf-8', ensure_ascii=False, indent=2, **kwargs):
    with open(json_file_path, 'w', encoding=encoding) as fout:
        json.dump(obj, fout,
                  ensure_ascii=ensure_ascii,
                  indent=indent,
                  **kwargs)


def default_load_pkl(pkl_file_path, **kwargs):
    with open(pkl_file_path, 'rb') as fin:
        obj = pickle.load(fin, **kwargs)

    return obj


def default_dump_pkl(obj, pkl_file_path, **kwargs):
    with open(pkl_file_path, 'wb') as fout:
        pickle.dump(obj, fout, **kwargs)


# def set_basic_log_config():
#     logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
#                         datefmt='%Y-%m-%d %H:%M:%S',
#                         level=logging.INFO)


class BERTChineseCharacterTokenizer(BertTokenizer):
    """
    Customized tokenizer for Chinese financial announcements
    
    Args:
        text (str): text to tokenize.
    """
    def tokenize(self, text):
        return list(text)

def recursive_print_grad_fn(grad_fn, prefix='', depth=0, max_depth=50):
    if depth > max_depth:
        return
    print(prefix, depth, grad_fn.__class__.__name__)
    if hasattr(grad_fn, 'next_functions'):
        for nf in grad_fn.next_functions:
            ngfn = nf[0]
            recursive_print_grad_fn(ngfn, prefix=prefix + '  ', depth=depth+1, max_depth=max_depth)

def strtobool(str_val):
    """Convert a string representation of truth to true (1) or false (0).

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    str_val = str_val.lower()
    if str_val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif str_val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError("invalid truth value %r" % (str_val,))


class InputExample(object):
    """
    A raw input example.
    """
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

class InputFeature(object):
    """
    A feature example to input model.
    """
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)