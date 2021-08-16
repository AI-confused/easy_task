# -*- coding: utf-8 -*-
# AUTHOR: Li Yun Liang
# DATE: 21-7-9


import os
import torch.optim as optim
import torch.distributed as dist
from itertools import product
import torch
from transformers import BertConfig
from tqdm import tqdm
from base.base_task import *
from base.base_setting import *
from base.base_utils import *
from .model import *
from .function import *


class CustomTask(BasePytorchTask):
    def __init__(self, task_setting: TaskSetting, load_train: bool=True, load_dev: bool=True, load_test: bool=True):
        """Custom Task definition class(custom).

        @task_setting: hyperparameters of Task.
        @load_train: load train set.
        @load_dev: load dev set.
        @load_test: load test set.
        """
        super(CustomTask, self).__init__(task_setting)
        self.logger.info('Initializing {}'.format(self.__class__.__name__))

        # prepare Model
        self.tokenizer = BERTChineseCharacterTokenizer.from_pretrained(self.setting.bert_model)
        self.bert_config = BertConfig.from_pretrained(self.setting.bert_model, num_labels=self.setting.num_label)
        self.setting.vocab_size = len(self.tokenizer.vocab)
        self.model = BertForSequenceClassification.from_pretrained(self.setting.bert_model, config=self.bert_config)
        self.decorate_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.setting.learning_rate))

        # load dataset
        self.load_data(read_examples, convert_examples_to_features, load_train, load_dev, load_test)

        # prepare custom batch convert func
        self.custom_collate_fn_train = train_collate_fn
        self.custom_collate_fn_eval = eval_collate_fn

        # best score and output result(custom)
        self.best_dev_score = 0.0
        self.best_test_score = 0.0
        self.output_result = {'result_type': '', 'task_config': self.setting.__dict__, 'result': []}


    def load_examples_features(self, load_example_func: function, convert_feature_func: function, data_type: str, file_name: str, flag: bool) -> tuple:
        """Load examples, features and dataset(custom).
        
        @load_example_func(func): read_examples
        @convert_feature_func(func): convert_examples_to_features
        @data_type: train or dev or test
        @file_name: dataset file name
        @flag: 1 means training, 0 means evaling
        """
        cached_features_file0 = os.path.join(self.setting.model_dir, 'cached_{}_{}_{}'.format(self.setting.percent, data_type, 'examples'))
        cached_features_file1 = os.path.join(self.setting.model_dir, 'cached_{}_{}_{}'.format(self.setting.percent, data_type, 'features'))

        if not self.setting.over_write_cache and os.path.exists(cached_features_file0) and os.path.exists(cached_features_file1):
            examples = torch.load(cached_features_file0)
            features = torch.load(cached_features_file1)
        else:
            examples = load_example_func(os.path.join(self.setting.data_dir, file_name), percent=self.setting.percent)
            features = convert_feature_func(examples,
                                            tokenizer=self.tokenizer,
                                            max_seq_len=self.setting.max_seq_len,
                                            is_training=flag)
 
            torch.save(examples, cached_features_file0)
            torch.save(features, cached_features_file1)
        dataset = TextDataset(features)
        return (examples, features, dataset)


    def train(self, resume_base_epoch=None):
        """Task level train func(custom)

        @resume_base_epoch(int): start training epoch
        """
        self.logger.info('=' * 20 + 'Start Training {}'.format(self.setting.task_name) + '=' * 20)

        # whether to resume latest cpt when restarting
        if resume_base_epoch is None:
            if self.setting.resume_latest_cpt:
                resume_base_epoch = self.get_latest_cpt_epoch()
            else:
                resume_base_epoch = 0

        # resume cpt if possible
        if resume_base_epoch > 0:
            self.logger.info('Training starts from epoch {}'.format(resume_base_epoch))
            self.resume_checkpoint(cpt_file_name='{}.cpt.{}'.format(self.setting.task_name, resume_base_epoch), resume_model=True, resume_optimizer=True)
        else:
            self.logger.info('Training starts from scratch')

        # do base train
        self.base_train(base_epoch_idx=resume_base_epoch)

        # save best score
        self.output_result['result'].append('best_dev_score: {} - best_test_score: {}'.format(self.best_dev_score, self.best_test_score))

        # write output results
        self.write_results()

    
    def eval(self, epoch):
        """Task level eval func(custom)

        @epoch(int): eval epoch
        """        
        for data_type in ['dev', 'test']:
            if data_type == 'test':
                features = self.test_features
                examples = self.test_examples
                dataset = self.test_dataset
            elif data_type == 'dev':
                features = self.dev_features
                examples = self.dev_examples
                dataset = self.dev_dataset

            # init Result class
            self.result = Result(task_name=self.setting.task_name)

            self.base_eval(epoch, data_type, examples, features, dataset)

            # calculate result score
            score = self.result.get_score()
            self.logger.info(score)
            
            # save best model with specific standard(custom)
            if data_type == 'dev' and score['f1_score'] > self.best_dev_score:
                self.best_dev_score = score['f1_score']
                self.logger.info('saving best dev model...')
                self.save_checkpoint(cpt_file_name='{}.cpt.{}.{}'.format(self.setting.task_name, data_type, 0))
                self.output_result['result'].append('data_type: {} - epoch: {} - train_loss: {} - epoch_score: {}'\
                                                    .format(data_type, epoch, self.train_loss, json.dumps(score, ensure_ascii=False)))

            if data_type == 'test' and score['f1_score'] > self.best_test_score:
                self.best_test_score = score['f1_score']
                self.logger.info('saving best test model...')
                self.save_checkpoint(cpt_file_name='{}.cpt.{}.{}'.format(self.setting.task_name, data_type, 0))
                self.output_result['result'].append('data_type: {} - epoch: {} - train_loss: {} - epoch_score: {}'\
                                                    .format(data_type, epoch, self.train_loss, json.dumps(score, ensure_ascii=False)))
                
            if self.setting.save_cpt_flag == 1:
                # save last epoch
                last_epoch = self.get_latest_cpt_epoch()
                if last_epoch != 0:
                    # delete lastest epoch model and store this epoch
                    delete_cpt_file = '{}.cpt.{}'.format(self.setting.task_name, last_epoch)
                    if os.path.exists(os.path.join(self.setting.model_dir, delete_cpt_file)):
                        os.remove(os.path.join(self.setting.model_dir, delete_cpt_file))
                    else:
                        self.logger.info("{} does not exist".format(delete_cpt_file), level=logging.WARNING)
                self.logger.info('saving latest epoch model...')
                self.save_checkpoint(cpt_file_name='{}.cpt.{}'.format(self.setting.task_name, epoch))
            elif self.setting.save_cpt_flag == 2:
                # save each epoch
                self.logger.info('saving epoch {} model...'.format(epoch))
                self.save_checkpoint(cpt_file_name='{}.cpt.{}'.format(self.setting.task_name, epoch))


    def get_result_on_batch(self, batch: tuple):
        """Return batch output logits during eval model(custom).

        @batch: /
        """
        input_ids, input_masks, segment_ids, labels = batch
        logits = self.model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_masks).detach().cpu()
        return logits, labels


    def get_loss_on_batch(self, batch):
        """Return batch loss during training model.

        @batch: /
        """
        input_ids, input_masks, segment_ids, labels = batch
        loss = self.model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_masks, labels=labels)
        return loss