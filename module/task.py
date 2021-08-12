# -*- coding: utf-8 -*-
# AUTHOR: Li Yun Liang
# DATE: 21-7-9


import yaml
import os
import torch.optim as optim
import torch.distributed as dist
from itertools import product
import json
import torch

from transformers import BertConfig
from tqdm import tqdm
from base.base_task import *
from base.base_setting import *
from base.base_utils import *
from .model import *
from .function import *
        


class CustomTask(BasePytorchTask):
    def __init__(self, task_setting, load_train=True, load_dev=True, load_test=True):
        """
        Retrospective approach event Extraction Task class
        Args:
            dee_setting (class TaskSetting): hyperparameters of Task.
            load_train (bool): load train set.
            load_dev (bool): load dev set.
            load_test (bool): load test set.
        """
        super(CustomTask, self).__init__(task_setting)
        self.logger.info('Initializing {}'.format(self.__class__.__name__))

        # prepare Model
        self.tokenizer = BERTChineseCharacterTokenizer.from_pretrained(self.setting.bert_model)
        self.bert_config = BertConfig.from_pretrained(self.setting.bert_model, num_labels=self.setting.num_label)
        self.setting.vocab_size = len(self.tokenizer.vocab)
        self.model = BertForSentenceClassification.from_pretrained(self.setting.bert_model, config=self.bert_config)
        self.decorate_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.setting.learning_rate))

        # load dataset
        self.load_data(read_examples, convert_examples_to_features, load_train, load_dev, load_test)

        # prepare custom batch convert func
        self.custom_collate_fn_train = train_collate_fn
        self.custom_collate_fn_eval = eval_collate_fn

        # task score & output file
        self.best_dev_score = 0.0
        self.best_test_score = 0.0
        self.output_file = os.path.join(self.setting.output_dir, 'result.json')


    def load_examples_features(self, load_example_func, convert_feature_func, data_type, file_name, flag):
        """
        load examples, features and dataset.(custom)
        Args:
            load_example_func(func): read_examples
            convert_feature_func(func): convert_examples_to_features

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
        return examples, features, dataset 


    def train(self, resume_base_epoch=None):
        """
        task level train func(custom)
        Args:
            resume_base_epoch(int): start training epoch
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
        self.base_train(
            base_epoch_idx=resume_base_epoch,      
        )

    
    def eval(self, epoch):
        """ 
        task level eval func(custom)
        Args:
            epoch(int): eval epoch
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
            
            # save best model
            if data_type == 'dev' and score['f1_score'] > self.best_dev_score:
                self.best_dev_score = score['f1_score']
                self.logger.info('saving best dev model...')
                self.save_checkpoint(cpt_file_name='{}.cpt.{}.{}'.format(self.setting.task_name, data_type, 0))

            if data_type == 'test' and score['f1_score'] > self.best_test_score:
                self.best_test_score = score['f1_score']
                self.logger.info('saving best test model...')
                self.save_checkpoint(cpt_file_name='{}.cpt.{}.{}'.format(self.setting.task_name, data_type, 0))
                
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



    def resume_eval_at_cpt(self, cpt_file_path):
        """
        load checkpoints and do eval

        Args:
            cpt_file_path:
        """
        # if self.is_master_node() and epoch >= 0:
        #     self.logger.info('\nPROGRESS: {}\n'.format(epoch / self.setting.num_train_epochs))

        # if resume_cpt_flag:
        # prepare model
        self.model = BertForSentenceClassification.from_pretrained(self.setting.bert_model, config=self.bert_config)
        # put model on cuda
        self.resume_checkpoint(cpt_file_path=cpt_file_path, resume_model=True, resume_optimizer=False)
        self._decorate_model()
        
        self.model_prefix = str(cpt_file_path).split('/')[-1]
        self.score_record[self.model_prefix] = {}
        self.eval(epoch=0, batch_size=self.setting.eval_batch_size)
        # if self.is_master_node() and save_cpt_flag:
        #     self.save_checkpoint(cpt_file_name='{}.cpt.{}'.format(self.setting.cpt_file_name, epoch), epoch=epoch)

        # if self.setting.model_type == 'classify':
        #     eval_tasks = ['dev', 'test']

        # for task_idx, data_type in enumerate(eval_tasks):
        #     if self.in_distributed_mode() and task_idx % dist.get_world_size() != dist.get_rank():
        #         continue

        #     if data_type == 'test':
        #         features = self.test_features
        #         examples = self.test_examples
        #         dataset = self.test_dataset
        #     elif data_type == 'dev':
        #         features = self.dev_features
        #         examples = self.dev_examples
        #         dataset = self.dev_dataset
        #     else:
        #         raise Exception('Unsupported data type {}'.format(data_type))

        #     # if gold_span_flag:
        #     #     span_str = 'gold_span'
        #     # else:
        #     #     span_str = 'pred_span'

        #     # if heuristic_type is None:
        #     #     # store user-provided name
        #     #     model_str = self.setting.cpt_file_name.replace('.', '~')
        #     # else:
        #     #     model_str = heuristic_type

        #     # decode_dump_name = decode_dump_template.format(data_type, span_str, model_str, epoch)
        #     # eval_dump_name = eval_dump_template.format(data_type, span_str, model_str, epoch)
        #     result = Result(task_name=self.setting.task_name)
        #     self.eval(self.setting.task_name, epoch, batch_size, examples, features, dataset, result)

        
    

    


    def get_result_on_batch(self, batch):
        """
        return batch output logits during eval model.
        """
        input_ids, input_masks, segment_ids, labels = batch
        logits = self.model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_masks).detach().cpu()
        return logits, labels


    def get_loss_on_batch(self, batch):
        """
        return batch loss during training model.
        """
        input_ids, input_masks, segment_ids, labels = batch
        loss = self.model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_masks, labels=labels)
        return loss