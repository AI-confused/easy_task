"""
-*- coding: utf-8 -*-
@author: black_tears
@time: 2021-07-09
@description: custom task file.
"""


import os
import random
from torch.jit import Error
import tqdm
import torch
import logging
import pandas as pd
from transformers import BertConfig, BertTokenizer
from base_module.base_task import *
from task_module.result_for_cls import *
from task_module.model_for_cls import *


class ClassificationTask(BasePytorchTask):
    def __init__(self, task_setting: TaskSetting, load_train: bool=False, load_dev: bool=False, load_test: bool=False):
        """Custom Task definition class(custom).

        @task_setting: hyperparameters of Task.
        @load_train: load train set.
        @load_dev: load dev set.
        @load_test: load test set.
        """
        super(ClassificationTask, self).__init__(task_setting)
        self.logger.info('Initializing {}'.format(self.__class__.__name__))

        # prepare model
        self.prepare_task_model()
        self._decorate_model()

        # prepare optim
        if load_train:
            self.prepare_optimizer()

        # load dataset
        self.load_data(load_train, load_dev, load_test)

        # best score and output result(custom)
        self.best_dev_score = 0.0
        self.best_dev_epoch = 0
        self.output_result = {'result_type': '', 'task_config': self.setting.__dict__, 'result': []}


    def prepare_task_model(self):
        """Prepare classification task model(custom).

        Can be overwriten.
        """
        self.tokenizer = BERTChineseCharacterTokenizer.from_pretrained(self.setting.bert_model)
        self.bert_config = BertConfig.from_pretrained(self.setting.bert_model, num_labels=self.setting.num_label)
        self.setting.vocab_size = len(self.tokenizer.vocab)
        self.model = BertForSequenceClassification.from_pretrained(self.setting.bert_model, config=self.bert_config)


    def prepare_optimizer(self):
        """Prepare cls task optimizer(custom).

        Can be overwriten.
        """
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.setting.learning_rate))


    def prepare_result_class(self):
        """Prepare result calculate class(custom).

        Can be overwriten.
        """
        self.result = ClassificationResult(task_name=self.setting.task_name)


    def load_examples_features(self, data_type: str, file_name: str, **kwargs) -> tuple:
        """Load examples, features and dataset(custom).

        Can be overwriten, but with the same input parameters and output type.
        
        @data_type: train or dev or test
        @file_name: dataset file name
        """
        cached_features_file0 = os.path.join(self.setting.model_dir, 'cached_{}_{}_{}'.format(self.setting.percent, data_type, 'examples'))
        cached_features_file1 = os.path.join(self.setting.model_dir, 'cached_{}_{}_{}'.format(self.setting.percent, data_type, 'features'))

        if not self.setting.over_write_cache and os.path.exists(cached_features_file0) and os.path.exists(cached_features_file1):
            examples = torch.load(cached_features_file0)
            features = torch.load(cached_features_file1)
        else:
            examples = self.read_examples(os.path.join(self.setting.data_dir, file_name), percent=self.setting.percent)
            torch.save(examples, cached_features_file0)
            features = self.convert_examples_to_features(examples,
                                                        tokenizer=self.tokenizer,
                                                        max_seq_len=self.setting.max_seq_len)
 
            torch.save(features, cached_features_file1)
        dataset = TextDataset(features)
        return (examples, features, dataset, features[0].max_seq_len)


    def read_examples(self, input_file: str, percent: float=1.0) -> list:
        """Read data from a data file and generate list of InputExamples(custom).

        Can be overwriten, but with the same input parameters and output type.

        @input_file: data input file abs dir
        @percent: percent of reading samples
        """
        examples=[]
        cnt = 10000

        try:
            data = json.load(open(input_file))['data']
        except:
            data = json.load(open(input_file))
            
        if percent != 1.0:
            data = random.sample(data, int(len(data)*percent))
        for line in tqdm.tqdm(data, desc='read examples'):
            text = line['text']
            try:
                # for data which has label
                label = line['label']
            except:
                # for data which don't have label
                label = -1
            doc_id = cnt
            cnt += 1
            examples.append(InputExample(
                doc_id=doc_id, 
                text=text,
                label=label))
        return examples


    def convert_examples_to_features(self, examples: list, tokenizer: BertTokenizer, max_seq_len: int, **kwargs) -> list:
        """Process the InputExamples into InputFeatures that can be fed into the model(custom).

        Can be overwriten, but with the same input parameters and output type.

        @examples: list of InputExamples
        @tokenizer: class BertTokenizer or its inherited classes
        @max_seq_len: max length of tokenized text
        """
        features = []
        for _ in tqdm.tqdm(range(len(examples)), total=len(examples), desc='convert features'):
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

            features.append(
                InputFeature(
                    doc_id=example.doc_id,
                    sentence=example.text,
                    input_tokens=input_token,
                    input_ids=input_id,
                    input_masks=input_mask,
                    segment_ids=segment_id,
                    sentence_len=sentence_len,
                    label=example.label,
                    max_seq_len=max_seq_len,
                )
            )
        return features


    def train(self, resume_base_epoch=None, resume_model_path=None):
        """Task level train func.

        @resume_base_epoch(int): start training epoch
        @resume_model_path(str): other model to restart with
        """
        self.logger.info('=' * 20 + 'Start Training {}'.format(self.setting.task_name) + '=' * 20)

        # resume model when restarting
        if resume_base_epoch is not None and resume_model_path is not None:
            raise ValueError('resume_base_epoch and resume_model_path can not be together!')
        elif resume_model_path is not None:
            self.logger.info('Training starts from other model: {}'.format(resume_model_path))
            self.resume_checkpoint(cpt_file_path=resume_model_path, resume_model=True, resume_optimizer=True)
            resume_base_epoch = 0
        else:
            if resume_base_epoch is None:
                if self.setting.resume_latest_cpt:
                    resume_base_epoch = self.get_latest_cpt_epoch()
                else:
                    resume_base_epoch = 0

            # resume cpt if possible
            if resume_base_epoch > 0:
                self.logger.info('Training starts from epoch {}'.format(resume_base_epoch))
                self.resume_checkpoint(cpt_file_name='{}.cpt.{}.e({}).b({}).p({}).s({})'.format(\
                    self.setting.task_name, resume_base_epoch, self.setting.num_train_epochs, self.setting.train_batch_size, str(self.setting.percent).replace('.','。'), self.setting.seed), resume_model=True, resume_optimizer=True)
            else:
                self.logger.info('Training starts from scratch')

        # prepare data loader
        self.train_dataloader = self._prepare_data_loader(self.train_dataset, self.setting.train_batch_size, rand_flag=True, collate_fn=self.custom_collate_fn_train)

        # do base train
        self._base_train(base_epoch_idx=resume_base_epoch)

        # save best score
        self.output_result['result'].append('best_dev_epoch: {} - best_dev_score: {}'.format(self.best_dev_epoch, self.best_dev_score))
        # self.output_result['result'].append('best_test_epoch: {} - best_test_score: {}'.format(self.best_test_epoch, self.best_test_score))

        # write output results
        self.write_results()

    
    def eval(self, epoch):
        """Task level eval func.

        @epoch(int): eval epoch
        """        
        data_type = 'dev'
        features = self.dev_features
        examples = self.dev_examples
        dataset = self.dev_dataset

        # prepare data loader
        self.eval_dataloader = self._prepare_data_loader(dataset, self.setting.eval_batch_size, rand_flag=False, collate_fn=self.custom_collate_fn_eval)

        # init result calculate class
        self.prepare_result_class()

        # do base eval
        self._base_eval(epoch, data_type, examples, features)

        # calculate result score
        score = self.result.get_score()
        self.logger.info(score)

        # return bad case in train-mode
        if self.setting.bad_case:
            self.return_selected_case(type_='badcase', items=self.result.bad_case, data_type=data_type, epoch=epoch)
        
        # save each epoch result
        self.output_result['result'].append('data_type: {} - epoch: {} - train_loss: {} - epoch_score: {}'\
                                            .format(data_type, epoch, self.train_loss, json.dumps(score, ensure_ascii=False)))

        # save best model with specific standard(custom)
        if data_type == 'dev' and score[self.setting.evaluation_metric] > self.best_dev_score:
            self.best_dev_epoch = epoch
            self.best_dev_score = score[self.setting.evaluation_metric]
            self.logger.info('saving best dev model...')
            self.save_checkpoint(cpt_file_name='{}.cpt.{}.{}.e({}).b({}).p({}).s({})'.format(\
                self.setting.task_name, data_type, 0, self.setting.num_train_epochs, self.setting.train_batch_size, str(self.setting.percent).replace('.','。'), self.setting.seed))
            
        save_cpt_file = '{}.cpt.{}.e({}).b({}).p({}).s({})'.format(\
                self.setting.task_name, epoch, self.setting.num_train_epochs, self.setting.train_batch_size, str(self.setting.percent).replace('.','。'), self.setting.seed)
        if self.setting.save_cpt_flag == 1 and not os.path.exists(os.path.join(self.setting.model_dir, save_cpt_file)):
            # save last epoch
            last_epoch = self.get_latest_cpt_epoch()
            if last_epoch != 0:
                # delete lastest epoch model and store this epoch
                delete_cpt_file = '{}.cpt.{}.e({}).b({}).p({}).s({})'.format(\
                    self.setting.task_name, last_epoch, self.setting.num_train_epochs, self.setting.train_batch_size, str(self.setting.percent).replace('.','。'), self.setting.seed)

                if os.path.exists(os.path.join(self.setting.model_dir, delete_cpt_file)):
                    os.remove(os.path.join(self.setting.model_dir, delete_cpt_file))
                    self.logger.info('remove model {}'.format(delete_cpt_file))
                else:
                    self.logger.info("{} does not exist".format(delete_cpt_file))

            self.logger.info('saving latest epoch model...')
            self.save_checkpoint(cpt_file_name='{}.cpt.{}.e({}).b({}).p({}).s({})'.format(\
                self.setting.task_name, epoch, self.setting.num_train_epochs, self.setting.train_batch_size, str(self.setting.percent).replace('.','。'), self.setting.seed))

        elif self.setting.save_cpt_flag == 2 and not os.path.exists(os.path.join(self.setting.model_dir, save_cpt_file)):
            # save each epoch
            self.logger.info('saving epoch {} model...'.format(epoch))
            self.save_checkpoint(cpt_file_name='{}.cpt.{}.e({}).b({}).p({}).s({})'.format(\
                self.setting.task_name, epoch, self.setting.num_train_epochs, self.setting.train_batch_size, str(self.setting.percent).replace('.','。'), self.setting.seed))


    def custom_collate_fn_train(self, features: list) -> list:
        """Convert batch training examples into batch tensor(custom).

        Can be overwriten, but with the same input parameters and output type.

        @examples(InputFeature): /
        """
        input_ids = torch.stack([torch.tensor(feature.input_ids, dtype=torch.long) for feature in features], 0)
        input_masks = torch.stack([torch.tensor(feature.input_masks, dtype=torch.long) for feature in features], 0)
        segment_ids = torch.stack([torch.tensor(feature.segment_ids, dtype=torch.long) for feature in features], 0)
        labels = torch.stack([torch.tensor(feature.label, dtype=torch.long) for feature in features], 0)

        return [input_ids, input_masks, segment_ids, labels, features]


    def custom_collate_fn_eval(self, features: list) -> list:
        """Convert batch eval examples into batch tensor(custom).

        Can be overwriten, but with the same input parameters and output type.

        @examples(InputFeature): /
        """
        input_ids = torch.stack([torch.tensor(feature.input_ids, dtype=torch.long) for feature in features], 0)
        input_masks = torch.stack([torch.tensor(feature.input_masks, dtype=torch.long) for feature in features], 0)
        segment_ids = torch.stack([torch.tensor(feature.segment_ids, dtype=torch.long) for feature in features], 0)
        labels = torch.stack([torch.tensor(feature.label, dtype=torch.long) for feature in features], 0)

        return [input_ids, input_masks, segment_ids, labels, features]


    def resume_test_at(self, resume_model_path: str, **kwargs):
        """Resume checkpoint and do test(custom).

        Can be overwriten, but with the same input parameters.
        
        @resume_model_path: do test model name
        """
        # extract kwargs
        header = kwargs.pop("header", True)

        self.resume_checkpoint(cpt_file_path=resume_model_path, resume_model=True, resume_optimizer=False)

        # prepare data loader
        self.eval_dataloader = self._prepare_data_loader(self.test_dataset, self.setting.eval_batch_size, rand_flag=False, collate_fn=self.custom_collate_fn_eval)

        # init result calculate class
        self.prepare_result_class()

        # do test
        self._base_eval(0, 'test', self.test_examples, self.test_features)

        # output test prediction
        self.return_selected_case(type_='prediction', items=self.result.prediction, file_type='csv', data_type='test', header=header)


    def get_result_on_batch(self, batch: tuple):
        """Return batch output logits during eval model(custom).

        Can be overwriten, but with the same input parameters and output type.

        @batch: /
        """
        input_ids, input_masks, segment_ids, labels, features = batch
        logits = self.model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_masks).detach().cpu()
        return logits, labels, features


    def get_loss_on_batch(self, batch):
        """Return batch loss during training model(custom).

        Can be overwriten, but with the same input parameters and output type.

        @batch: /
        """
        input_ids, input_masks, segment_ids, labels, features = batch
        loss = self.model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_masks, labels=labels)
        return loss
        