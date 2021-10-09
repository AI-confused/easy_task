"""
-*- coding: utf-8 -*-
@author: black_tears
@time: 2021-07-09
@description: custom task file.
"""


import os
import random
import tqdm
import torch
import logging
import pandas as pd
from transformers import BertConfig, BertTokenizer
# from base.base_task import BasePytorchTask
# from base.base_setting import TaskSetting
from base import *
from module import *
# from .model_for_cls import BertForSequenceClassification
# from .utils_for_cls import *


class ClassificationTask(BasePytorchTask):
    def __init__(self, task_setting: TaskSetting, load_train: bool=True, load_dev: bool=True, load_test: bool=True):
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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.setting.learning_rate))

        # load dataset
        self.load_data(load_train, load_dev, load_test)

        # best score and output result(custom)
        self.best_dev_score = 0.0
        self.best_test_score = 0.0
        self.output_result = {'result_type': '', 'task_config': self.setting.__dict__, 'result': []}


    def prepare_task_model(self):
        """Prepare classification task model(custom).

        Can be overwriten.
        """
        self.tokenizer = BertTokenizer.from_pretrained(self.setting.bert_model)
        self.bert_config = BertConfig.from_pretrained(self.setting.bert_model, num_labels=self.setting.num_label)
        self.setting.vocab_size = len(self.tokenizer.vocab)
        self.model = BertForSequenceClassification.from_pretrained(self.setting.bert_model, config=self.bert_config)


    def load_examples_features(self, data_type: str, file_name: str, flag: bool) -> tuple:
        """Load examples, features and dataset(custom).

        Can be overwriten, but with the same input parameters and output type.
        
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
            examples = self.read_examples(os.path.join(self.setting.data_dir, file_name), percent=self.setting.percent)
            features = self.convert_examples_to_features(examples,
                                                        tokenizer=self.tokenizer,
                                                        max_seq_len=self.setting.max_seq_len,
                                                        is_training=flag)
 
            torch.save(examples, cached_features_file0)
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


    def convert_examples_to_features(self, examples: list, tokenizer: BertTokenizer, max_seq_len: int, **kwargs) -> list:
        """Process the InputExamples into InputFeatures that can be fed into the model(custom).

        Can be overwriten, but with the same input parameters and output type.

        @examples: list of InputExamples
        @tokenizer: class BertTokenizer or its inherited classes
        @max_seq_len: max length of tokenized text
        """
        features = []
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

            features.append(
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
        return features


    def train(self, resume_base_epoch=None):
        """Task level train func.

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

        # prepare data loader
        self.train_dataloader = self._prepare_data_loader(self.train_dataset, self.setting.train_batch_size, rand_flag=True, collate_fn=self.custom_collate_fn_train)

        # do base train
        self._base_train(base_epoch_idx=resume_base_epoch)

        # save best score
        self.output_result['result'].append('best_dev_score: {} - best_test_score: {}'.format(self.best_dev_score, self.best_test_score))

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

            # init Result class
            self.result = ClassificationResult(task_name=self.setting.task_name)

            # prepare data loader
            self.eval_dataloader = self._prepare_data_loader(dataset, self.setting.eval_batch_size, rand_flag=False, collate_fn=self.custom_collate_fn_eval)

            self._base_eval(epoch, data_type, examples, features)

            # calculate result score
            score = self.result.get_score()
            self.logger.info(score)

            # return bad case in train-mode
            if self.setting.train_bad_case:
                self.return_bad_case(data_type=data_type, epoch=epoch)
            
            # save each epoch result
            self.output_result['result'].append('data_type: {} - epoch: {} - train_loss: {} - epoch_score: {}'\
                                                .format(data_type, epoch, self.train_loss, json.dumps(score, ensure_ascii=False)))

            # save best model with specific standard(custom)
            if data_type == 'dev' and score['f1_score'][1] > self.best_dev_score:
                self.best_dev_score = score['f1_score'][1]
                self.logger.info('saving best dev model...')
                self.save_checkpoint(cpt_file_name='{}.cpt.{}.{}'.format(self.setting.task_name, data_type, 0))

            if data_type == 'test' and score['f1_score'][1] > self.best_test_score:
                self.best_test_score = score['f1_score'][1]
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


    def resume_eval_at(self, resume_model_name: str):
        """Resume checkpoint and do eval(custom).

        Can be overwriten, but with the same input parameters.
        
        @resume_model_dir: do test model name
        """
        self.resume_checkpoint(cpt_file_name=resume_model_name, resume_model=True, resume_optimizer=False)

        # init Result class
        self.result = ClassificationResult(task_name=self.setting.task_name)

        # prepare data loader
        self.eval_dataloader = self._prepare_data_loader(self.test_dataset, self.setting.eval_batch_size, rand_flag=False, collate_fn=self.custom_collate_fn_eval)

        # do test
        self._base_eval(0, 'test', self.test_examples, self.test_features)

        # calculate result score
        score = self.result.get_score()
        self.logger.info(score)

        # write results
        self.output_result['result'].append('test_score: {}'.format(json.dumps(score, ensure_ascii=False)))

        # write output results
        self.write_results()

        # write bad case
        if self.setting.test_bad_case:
            self.return_bad_case()
    

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


    def write_results(self):
        """Write results to output file.

        Can be overwriten.
        """
        result_file = os.path.join(self.setting.result_dir, 'result-{}-{}.json'.format(self.now_time, self.output_result['result_type']))

        # add result_type: train or test
        BaseUtils.write_lines(file_path=result_file, content=[self.output_result['result_type']], write_type='w')
        BaseUtils.write_lines(file_path=result_file, content=['*'*40])

        # add task configuration
        for key, value in self.output_result['task_config'].items():
            BaseUtils.write_lines(file_path=result_file, content=['{}: {}'.format(key, value)])
        BaseUtils.write_lines(file_path=result_file, content=['*'*40])

        # add each epoch eval result or test result
        BaseUtils.write_lines(file_path=result_file, content=self.output_result['result'])
        self.logger.info('write results to {}'.format(result_file))

        
    def return_bad_case(self, file_type: str='excel', data_type: str='', epoch=''):
        """Return eval bad case and dump to file.

        Can be overwriten.

        @file_type: file type of bad case.
        @data_type: test or dev during train-mode, not used during test-mode.
        epoch: train epoch during train-mode, not used during test-mode.
        """
        dataframe = pd.DataFrame(self.result.bad_case)
        if file_type == 'excel':
            bad_case_file = os.path.join(self.setting.result_dir, 'badcase-{}-{}-{}-{}.xlsx'.format(self.now_time, data_type, epoch, self.output_result['result_type']))
            dataframe.to_excel(bad_case_file, index=False)
        elif file_type == 'csv':
            bad_case_file = os.path.join(self.setting.result_dir, 'badcase-{}-{}-{}-{}.csv'.format(self.now_time, data_type, epoch, self.output_result['result_type']))
            dataframe.to_csv(bad_case_file, index=False)
        else:
            raise ValueError('Wrong file_type!')

        self.logger.info('write badcases to {}'.format(bad_case_file))
