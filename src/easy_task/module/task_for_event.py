"""
-*- coding: utf-8 -*-
@author: black_tears
@time: 2021-10-26
@description: custom event task file.
"""


import os
import random
import logging
import tqdm
import torch
import pandas as pd
from transformers import BertConfig
from module.task_for_ner import *
from module.result_for_event import *
from module.model_for_event import *


class EventExtractionTask(SequenceTaggingTask):
    def __init__(self, task_setting: TaskSetting, load_train: bool=False, load_dev: bool=False, load_test: bool=False):
        """Custom Task definition class(custom).

        @task_setting: hyperparameters of Task.
        @load_train: load train set.
        @load_dev: load dev set.
        @load_test: load test set.
        """
        super(EventExtractionTask, self).__init__(task_setting, load_train=load_train, load_dev=load_dev, load_test=load_test)
        pass


    def prepare_result_class(self):
        """Prepare result calculate class(custom).

        Can be overwriten.
        """
        self.result = EventExtractionResult(task_name=self.setting.task_name, id2label=self.setting.id2label, max_seq_len=self.setting.max_seq_len)


    def read_examples(self, input_file: str, percent: float=1.0) -> tuple:
        """Read data from a data file and generate list of InputExamples(custom).

        Can be overwriten, but with the same input parameters and output type.

        @input_file: data input file abs dir
        @percent: percent of reading samples
        """
        examples=[]

        data = pd.read_csv(input_file)
            
        if percent != 1.0:
            data = data.sample(frac=percent, random_state=self.setting.seed)
            data = data.reset_index(drop=True)

        doc_arguments_label = {}
        cnt = 0

        for val in tqdm.tqdm(data[['content','doc_id', 'event_type','arguments']].values, desc='read examples'):
            examples.append(InputExample(
                doc_id=str(val[1]), 
                text=val[0],
                event_type = json.loads(val[2]),
                # event_type_label = json.loads(val[3]),
                arguments = json.loads(val[-1]),
            ))
            doc_arguments_label[str(val[1])] = {}
            doc_arguments_label[str(val[1])]['event_type'] = json.loads(val[2])
            doc_arguments_label[str(val[1])]['arguments'] = json.loads(val[-1])
            cnt += 1

        return (examples, doc_arguments_label)


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

            if example.label is not None:
                # tag label
                labels = sorted(example.label, key=lambda x: x[1])
                sen_labels = []
                last_point = 0
                for _ in labels:
                    if _[1] < last_point:
                        continue
                    sen_labels += [0]*(_[1]-last_point)
                    sen_labels += [self.setting.label2id[_[0]]]
                    sen_labels += [self.setting.label2id[_[0]] + (len(self.setting.label2id) - 1)] * (_[2] - _[1] - 1)
                    last_point = _[2]
                sen_labels += [0] * (len(example.text) - last_point)
                assert len(example.text) == len(sen_labels)

                # tokenize
                sentence_token = tokenizer.tokenize(example.text)[:max_seq_len-2]
                example.text = example.text[:max_seq_len-2]
                sen_labels = sen_labels[:max_seq_len-2]
                assert len(sentence_token) == len(sen_labels)
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
                BIO_label = [0] + sen_labels + [-1]*padding_length + [0]
                assert len(BIO_label) == max_seq_len

                features.append(
                    InputFeature(
                        doc_id=example.doc_id,
                        sentence=example.text,
                        entity_label=labels,
                        input_tokens=input_token,
                        input_ids=input_id,
                        input_masks=input_mask,
                        segment_ids=segment_id,
                        sentence_len=sentence_len,
                        label=BIO_label,
                        max_seq_len=max_seq_len
                    )
                )
            else:
                # tokenize
                sentence_token = tokenizer.tokenize(example.text)[:max_seq_len-2]
                example.text = example.text[:max_seq_len-2]
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
                        max_seq_len=max_seq_len
                    )
                )

        return features


    def train(self, resume_base_epoch: int=None, resume_model_path: str=None):
        """Task level train func.

        @resume_base_epoch(int): start training epoch
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
                self.resume_checkpoint(cpt_file_name='{}.cpt.{}'.format(self.setting.task_name, resume_base_epoch), resume_model=True, resume_optimizer=True)
            else:
                self.logger.info('Training starts from scratch')

        # prepare data loader
        self.train_dataloader = self._prepare_data_loader(self.train_dataset, self.setting.train_batch_size, rand_flag=True, collate_fn=self.custom_collate_fn_train)

        # do base train
        self._base_train(base_epoch_idx=resume_base_epoch)

        # save best score
        self.output_result['result'].append('best_dev_epoch: {} - best_dev_score: {}'.format(self.best_dev_epoch, self.best_dev_score))
        self.output_result['result'].append('best_test_epoch: {} - best_test_score: {}'.format(self.best_test_epoch, self.best_test_score))

        # write output results
        self.write_results()

    
    def eval(self, epoch):
        """Task level eval func.

        @epoch(int): eval epoch
        """        
        for data_type in self.setting.eval_file:
            if data_type == 'test':
                features = self.test_features
                examples = self.test_examples
                dataset = self.test_dataset
            elif data_type == 'dev':
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
                self.return_selected_case(type_='bad_case', items=self.result.bad_case, data_type=data_type, epoch=epoch)

            # return all result
            self.return_selected_case(type_='eval_prediction', items=self.result.all_result, data_type=data_type, epoch=epoch)
            
            # save each epoch result
            self.output_result['result'].append('data_type: {} - epoch: {} - train_loss: {} - epoch_score: {}'\
                                                .format(data_type, epoch, self.train_loss, json.dumps(score, ensure_ascii=False)))

            # save best model with specific standard(custom)
            if data_type == 'dev' and score[self.setting.evaluation_metric] > self.best_dev_score:
                self.best_dev_epoch = epoch
                self.best_dev_score = score[self.setting.evaluation_metric]
                self.logger.info('saving best dev model...')
                self.save_checkpoint(cpt_file_name='{}.cpt.{}.{}.e{}.b{}.p{}.s{}'.format(\
                    self.setting.task_name, data_type, 0, self.setting.num_train_epochs, self.setting.train_batch_size, self.setting.percent, self.setting.seed))

            if data_type == 'test' and score[self.setting.evaluation_metric] > self.best_test_score:
                self.best_test_epoch = epoch
                self.best_test_score = score[self.setting.evaluation_metric]
                self.logger.info('saving best test model...')
                self.save_checkpoint(cpt_file_name='{}.cpt.{}.{}.e{}.b{}.p{}.s{}'.format(\
                    self.setting.task_name, data_type, 0, self.setting.num_train_epochs, self.setting.train_batch_size, self.setting.percent, self.setting.seed))
                
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
                self.save_checkpoint(cpt_file_name='{}.cpt.{}.{}.e{}.b{}.p{}.s{}'.format(\
                    self.setting.task_name, data_type, epoch, self.setting.num_train_epochs, self.setting.train_batch_size, self.setting.percent, self.setting.seed))
            elif self.setting.save_cpt_flag == 2:
                # save each epoch
                self.logger.info('saving epoch {} model...'.format(epoch))
                self.save_checkpoint(cpt_file_name='{}.cpt.{}.{}.e{}.b{}.p{}.s{}'.format(\
                    self.setting.task_name, data_type, epoch, self.setting.num_train_epochs, self.setting.train_batch_size, self.setting.percent, self.setting.seed))


    def custom_collate_fn_train(self, features: list) -> list:
        """Convert batch training examples into batch tensor(custom).

        Can be overwriten, but with the same input parameters and output type.

        @examples(InputFeature): /
        """
        input_ids = torch.stack([torch.tensor(feature.input_ids, dtype=torch.long) for feature in features], 0)
        input_masks = torch.stack([torch.tensor(feature.input_masks, dtype=torch.long) for feature in features], 0)
        segment_ids = torch.stack([torch.tensor(feature.segment_ids, dtype=torch.long) for feature in features], 0)
        labels = torch.stack([torch.tensor(feature.label, dtype=torch.long) for feature in features], 0)

        return [input_ids, input_masks, segment_ids, labels]


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


    def resume_test_at(self, resume_model_path: str):
        """Resume checkpoint and do test(custom).

        Can be overwriten, but with the same input parameters.
        
        @resume_model_path: do test model path
        """
        self.resume_checkpoint(cpt_file_path=resume_model_path, resume_model=True, resume_optimizer=False)

        # prepare data loader
        self.eval_dataloader = self._prepare_data_loader(self.test_dataset, self.setting.eval_batch_size, rand_flag=False, collate_fn=self.custom_collate_fn_eval)

        # init result calculate class
        self.prepare_result_class()

        # do test
        self._base_eval(0, 'test', self.test_examples, self.test_features)

        # output test prediction
        self.result.get_prediction()
        self.return_selected_case(type_='test_prediction', items=self.result.all_result)
    

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
        input_ids, input_masks, segment_ids, labels = batch
        loss = self.model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_masks, labels=labels)
        return loss
