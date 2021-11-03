"""
-*- coding: utf-8 -*-
@author: black_tears
@time: 2021-07-09
@description: base module of task definition.
"""


import logging
from pickle import NONE
import random
from datetime import datetime
import pandas as pd
import os
import abc
import json
import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.nn.parallel as para
from base.base_setting import *
from base.base_utils import *


class BasePytorchTask(metaclass=abc.ABCMeta):
    def __init__(self, setting: TaskSetting):
        """Basic task to support deep learning models on Pytorch.

        Custom pytorch task should inherit this class and input task_setting.

        @setting: hyperparameters of Pytorch Task.
        """
        self.setting = setting
                
        # general initialization
        self.__set_basic_log_config()
        self.__check_setting_validity()
        self.__init_device()
        self.reset_random_seed()

        # task-specific initialization
        self.train_examples = None
        self.train_features = None
        self.train_dataset = None
        self.dev_examples = None
        self.dev_features = None
        self.dev_dataset = None
        self.test_examples = None
        self.test_features = None
        self.test_dataset = None
        self.train_dataloader = None
        self.eval_dataloader = None
        self.model = None
        self.optimizer = None
        self.num_train_steps = None
        self.model_named_parameters = None
        self.best_dev_score = None
        self.best_test_score = None
        self.output_result = None
        self.eval_best_loss = 1000
        self.early_stop_flag = 0


    def __set_basic_log_config(self):
        """Set basic logger configuration.

        Private class function.
        """
        # get now time
        self.now_time = datetime.now().strftime("%Y-%m-%d|%H:%M:%S")

        # set logging format
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s', 
                                        datefmt='%Y-%m-%d | %H:%M:%S')
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        # output to log file handler
        file_handler = logging.FileHandler(os.path.join(self.setting.log_dir, 'log-{}.log'.format(self.now_time)))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        # output to cmd
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)

        # add handler
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)


    def __check_setting_validity(self):
        """Check task setting parameters are valid or not.

        Private class function.
        """
        self.logger.info('='*20 + 'Check Setting Validity' + '='*20)
        self.logger.info('Setting: {}'.format(
            json.dumps(self.setting.__dict__, ensure_ascii=False, indent=2)
        ))

        # check valid grad accumulate step
        if self.setting.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                self.setting.gradient_accumulation_steps))
        # set concrete train batch size according to gradient_accumulation_steps
        self.setting.train_batch_size = int(self.setting.train_batch_size / self.setting.gradient_accumulation_steps)
    
        
    def __init_device(self):
        """Init device.

        Private class function.
        """
        self.logger.info('='*20 + 'Init Device' + '='*20)
        
        # set device
        os.environ["CUDA_VISIBLE_DEVICES"] = self.setting.cuda_device
        self.device = torch.device("cuda" if torch.cuda.is_available() and not self.setting.no_cuda else "cpu")
        if not self.setting.no_cuda:
            self.n_gpu = torch.cuda.device_count()
        else:
            self.n_gpu = 0
        self.logger.info("device {} / n_gpu {}".format(self.device, self.n_gpu))


    def __set_batch_to_device(self, batch: list):
        """Put batch data into device.

        @batch: batch features on device
        """
        res = []
        for x in batch:
            if isinstance(x, torch.Tensor):
                x = x.to(self.device)
                res.append(x)
            else:
                res.append(x)
        return res


    def _decorate_model(self):
        """Put model on device.
        """
        self.logger.info('='*20 + 'Decorate Model' + '='*20)
        
        if self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
            self.logger.info('Multi-gpu task')
        elif self.n_gpu == 1:
            self.logger.info('Single-gpu task')
        else:
            self.logger.info('no cuda task')
        
        self.model.to(self.device)
        self.logger.info('Set model device to {}'.format(str(self.device)))


    def get_latest_cpt_epoch(self) -> int:
        """Get the latest training epoch model from the saved model directory.
        """
        prev_epochs = []

        # find checkpoints and sort them by epoch
        for fn in os.listdir(self.setting.model_dir):
            if fn.startswith('{}.cpt'.format(self.setting.task_name)):
                try:
                    epoch = int(fn.split('.')[-5])
                    if epoch > 0:
                        prev_epochs.append(epoch)
                except Exception as e:
                    continue
        prev_epochs.sort()

        if len(prev_epochs) > 0:
            latest_epoch = prev_epochs[-1]
            self.logger.info('Pick latest epoch {} from {}'.format(latest_epoch, str(prev_epochs)))
        else:
            latest_epoch = 0
            self.logger.info('No previous epoch checkpoints, just start from scratch')

        return latest_epoch
        

    def reset_random_seed(self, seed: int=None):
        """Reset random seed during task.

        @seed: random seed.
        """
        if seed is None:
            seed = self.setting.seed
        self.logger.info('='*20 + 'Reset Random Seed to {}'.format(seed) + '='*20)

        # set random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(seed)


    def _prepare_data_loader(self, dataset: list, batch_size: int, rand_flag: bool=True, collate_fn=None):
        """Prepare dataloader during task.

        @dataset: list of InputFeature
        @batch_size: train or eval batch
        @rand_flag: use RandomSampler or not
        @collate_fun of deposing batch data to tensor
        """
        if rand_flag:
            data_sampler = RandomSampler(dataset)
        else:
            data_sampler = SequentialSampler(dataset)

        if collate_fn is not None:
            dataloader = DataLoader(dataset,
                                    batch_size=batch_size,
                                    sampler=data_sampler,
                                    collate_fn=collate_fn)
        else:
            dataloader = DataLoader(dataset,
                                    batch_size=batch_size,
                                    sampler=data_sampler)

        return dataloader


    def load_data(self, load_train: bool, load_dev: bool, load_test: bool, **kwargs):
        """Load dataset and construct model's examples, features and dataset.

        @load_train: load train portion or not
        @load_dev: load dev portion or not
        @load_test: load test portion or not
        """
        self.logger.info('='*20 + 'load dataset' + '='*20)

        #load train portion
        if load_train:
            self.logger.info('Load train portion')
            self.train_examples, self.train_features, self.train_dataset, self.setting.max_seq_len = self.load_examples_features('train', self.setting.train_file_name)
            self.logger.info('Load train portion done')
            self.logger.info('training examples: {}, features: {} at max_sequence_len: {}'.format(len(self.train_examples), len(self.train_features), self.setting.max_seq_len))
        else:
            self.logger.info('Do not load train portion')

        # load dev portion
        if load_dev:
            self.logger.info('Load dev portion')
            self.dev_examples, self.dev_features, self.dev_dataset, self.setting.max_seq_len = self.load_examples_features('dev', self.setting.dev_file_name)
            self.logger.info('Load dev portion done!')
            self.logger.info('dev examples: {}, features: {} at max_sequence_len: {}'.format(len(self.dev_examples), len(self.dev_features), self.setting.max_seq_len))
        else:
            self.logger.info('Do not load dev portion')

        # load test portion
        if load_test:
            self.logger.info('Load test portion')
            self.test_examples, self.test_features, self.test_dataset, self.setting.max_seq_len = self.load_examples_features('test', self.setting.test_file_name)
            self.logger.info('Load test portion done!')
            self.logger.info('test examples: {}, features: {} at max_sequence_len: {}'.format(len(self.test_examples), len(self.test_features), self.setting.max_seq_len))
        else:
            self.logger.info('Do not load test portion')    


    def _base_train(self, **kwargs):
        """Base task train func with a set of parameters.

        This class should not be rewritten.
        
        @kwargs: base_epoch_idx
        """
        assert self.model is not None
        if self.num_train_steps is None:
            self.num_train_steps = self.setting.num_train_epochs * len(self.train_features) // self.setting.train_batch_size

        self.logger.info('='*20 + 'Start Base Training' + '='*20)
        self.logger.info("\tTotal examples Num = {}".format(len(self.train_examples)))
        self.logger.info("\tTotal features Num = {}".format(len(self.train_features)))
        self.logger.info("\tBatch size = {}".format(self.setting.train_batch_size))
        self.logger.info("\tNum steps = {}".format(self.num_train_steps))     

        # start training
        global_step = 0
        self.train_loss = 0

        for epoch_idx in tqdm.trange(kwargs['base_epoch_idx'], int(self.setting.num_train_epochs), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            bar = tqdm.tqdm(self.train_dataloader)
            for step, batch in enumerate(bar):
                batch = self.__set_batch_to_device(batch)
                loss = self.get_loss_on_batch(batch)

                if self.n_gpu > 1:
                    # mean() to average on multi-gpu.
                    loss = loss.mean()  
                if self.setting.gradient_accumulation_steps > 1:
                    loss = loss / self.setting.gradient_accumulation_steps

                # backward
                loss.backward()

                loss_scalar = loss.item()
                tr_loss += loss_scalar
                self.train_loss = round(tr_loss * self.setting.gradient_accumulation_steps / (nb_tr_steps+1), 4)
                bar.set_description('loss {}'.format(self.train_loss))
                nb_tr_examples += self.setting.train_batch_size
                nb_tr_steps += 1

                if (step + 1) % self.setting.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.model.zero_grad()
                    global_step += 1
            # do epoch eval
            self.eval(epoch_idx + 1)
            # do early stop
            if hasattr(self.setting, 'do_early_stop') and self.setting.do_early_stop and self.early_stopping():
                self.logger.info('='*20 + 'Early Stop Base Training' + '='*20)
                break


    def _base_eval(self, epoch: int, data_type: str, eval_examples: list, eval_features: list, **kwargs):
        """Base task eval func with a set of parameters.

        This class should not be rewritten.

        @epoch: eval epoch
        @data_type: 'dev' or 'test'
        @eval_examples: list of InputExample
        @eval_features: list of InputFeature
        """
        assert self.model is not None
        self.logger.info('=' * 20 + 'Start Evaluation/{}'.format(data_type) + '=' * 20)
        self.logger.info('\tPROGRESS: {}'.format(epoch / self.setting.num_train_epochs))
        self.logger.info("\tNum examples = {}".format(len(eval_examples)))
        self.logger.info("\tNum features = {}".format(len(eval_features)))
        self.logger.info("\tBatch size = {}".format(self.setting.eval_batch_size))

        # enter eval mode
        self.model.eval()

        # do eval
        eval_steps, eval_loss = 0, 0.0
        for batch in tqdm.tqdm(self.eval_dataloader, desc='Iteration'):
            batch = self.__set_batch_to_device(batch)

            with torch.no_grad():
                batch_eval_loss = self.get_loss_on_batch(batch)
                batch_results = self.get_result_on_batch(batch)
                self.result.update_batch(batch_results=batch_results)

            if self.n_gpu > 1:
                # mean() to average on multi-gpu.
                batch_eval_loss = batch_eval_loss.mean()  
            if self.setting.gradient_accumulation_steps > 1:
                batch_eval_loss = batch_eval_loss / self.setting.gradient_accumulation_steps

            eval_loss += batch_eval_loss.item()
            eval_steps += 1
        # calculate epoch eval loss
        self.eval_loss = eval_loss / eval_steps
        self.logger.info("\tEpoch Eval Loss = {}".format(self.eval_loss))


    def early_stopping(self):
        """Do early stop during epoch training.
        """
        if self.eval_loss < self.eval_best_loss:
            self.early_stop_flag = 0
            self.eval_best_loss = self.eval_loss
        else:
            self.early_stop_flag += 1
            # stop training
            if self.early_stop_flag == 2:
                return 1
            # update learning rate of parameters of model
            for params in self.optimizer.param_groups:  
                params['lr'] *= 0.5
        return 0


    def save_checkpoint(self, cpt_file_name: str=None, epoch: int=None):
        """Save save_checkpoint file to model path.

        @cpt_file_name: saved file name.
        @epoch: num of saving model epoch.
        """
        self.logger.info('='*20 + 'Dump Checkpoint' + '='*20)
        if cpt_file_name is None:
            cpt_file_name = self.setting.cpt_file_name
        cpt_file_path = os.path.join(self.setting.model_dir, cpt_file_name)
        self.logger.info('Dump checkpoint into {}'.format(cpt_file_path))

        store_dict = {
            'setting': self.setting.__dict__,
        }

        # save model parameters
        if self.model:
            if isinstance(self.model, para.DataParallel) or \
                    isinstance(self.model, para.DistributedDataParallel):
                model_state = self.model.module.state_dict()
            else:
                model_state = self.model.state_dict()
            store_dict['model_state'] = model_state
        else:
            self.logger.info('No model state is dumped', level=logging.WARNING)

        # save optimizer parameters
        if self.optimizer:
            store_dict['optimizer_state'] = self.optimizer.state_dict()
        else:
            self.logger.info('No optimizer state is dumped', level=logging.WARNING)

        if epoch:
            store_dict['epoch'] = epoch

        torch.save(store_dict, cpt_file_path)


    def resume_checkpoint(self, cpt_file_name: str=None, cpt_file_path: str=None, resume_model: bool=True, resume_optimizer: bool=False, strict: bool=False):
        """Load checkpoint from saved file.

        @cpt_file_name: saved model file name.
        @cpt_file_path: abs path of resume model file.
        @resume_model: load model weights.
        @resume_optimizer: load optimizer weights.
        @strict: /
        """
        self.logger.info('='*20 + 'Resume Checkpoint' + '='*20)

        # build checkpoint file path
        if (cpt_file_name is not None and cpt_file_path is not None) or (cpt_file_name is None and cpt_file_path is None):
            raise ValueError('cpt_file_name and cpt_file_path must have one!')
        elif cpt_file_name is not None:
            cpt_file_path = os.path.join(self.setting.model_dir, cpt_file_name)
        
        if os.path.exists(cpt_file_path):
            self.logger.info('Resume checkpoint from {}'.format(cpt_file_path))
        else:
            self.logger.warning('Checkpoint {} does not exist'.format(cpt_file_path))
            raise Exception('Resume checkpoint failed')

        if torch.cuda.device_count() == 0:
            store_dict = torch.load(cpt_file_path, map_location='cpu')
        else:
            store_dict = torch.load(cpt_file_path, map_location=self.device)

        if resume_model:
            if self.model and 'model_state' in store_dict:
                if isinstance(self.model, para.DataParallel) or \
                        isinstance(self.model, para.DistributedDataParallel):
                    self.model.module.load_state_dict(store_dict['model_state'])
                else:
                    self.model.load_state_dict(store_dict['model_state'])
                self.logger.info('Resume model successfully')
            elif strict:
                raise Exception('Resume model failed, dict.keys = {}'.format(store_dict.keys()))
        else:
            self.logger.info('Do not resume model')

        if resume_optimizer:
            if self.optimizer and 'optimizer_state' in store_dict:
                self.optimizer.load_state_dict(store_dict['optimizer_state'])
                self.logger.info('Resume optimizer successfully')
            elif strict:
                raise Exception('Resume optimizer failed, dict.keys = {}'.format(store_dict.keys()))
        else:
            self.logger.info('Do not resume optimizer')


    def return_selected_case(self, type_: str, items: dict, **kwargs):
        """Return eval selected case and dump to file.

        Can be overwritten.

        @type_: bad_case or prediction.
        @items: {'text': [], 'id': [], 'pred': [], 'label': []}
        @file_type: file type of bad case.
        @data_type: dev or test.
        @epoch: train epoch in train-mode, not used in test-mode.
        @header: pandas dump to file whether has columns.
        """
        # extract kwargs
        file_type = kwargs.pop('file_type', 'csv')
        data_type = kwargs.pop('data_type', '')
        epoch = kwargs.pop('epoch', '')
        header = kwargs.pop('header', True)

        dataframe = pd.DataFrame(items)
        if file_type == 'excel':
            bad_case_file = os.path.join(self.setting.result_dir, '{}-{}-{}-{}-{}.xlsx'.format(type_, self.now_time, data_type, epoch, self.output_result['result_type']))
            dataframe.to_excel(bad_case_file, index=False, header=header)
        elif file_type == 'csv':
            bad_case_file = os.path.join(self.setting.result_dir, '{}-{}-{}-{}-{}.csv'.format(type_, self.now_time, data_type, epoch, self.output_result['result_type']))
            dataframe.to_csv(bad_case_file, index=False, header=header)
        else:
            raise ValueError('Wrong file_type!')

        self.logger.info('write {} to {}'.format(type_, bad_case_file))


    def write_results(self):
        """Write results to output file.

        Can be overwritten.
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


    def custom_collate_fn_train(self, examples: list) -> list:
        """Convert batch training examples into batch tensor.

        Should be written by inherit class.

        @examples(InputFeature): /
        """
        pass


    def custom_collate_fn_eval(self, examples: list) -> list:
        """Convert batch eval examples into batch tensor.

        Should be written by inherit class.

        @examples(InputFeature): /
        """
        pass


    @abc.abstractclassmethod
    def prepare_task_model(self):
        """Prepare task model.

        Must be written by inherit class.
        """
        pass


    @abc.abstractclassmethod
    def prepare_optimizer(self):
        """repare task optimizer(custom).

        Must be written by inherit class.
        """
        pass


    @abc.abstractclassmethod
    def prepare_result_class(self):
        """repare task result calculate class(custom).

        Must be written by inherit class.
        """
        pass


    @abc.abstractclassmethod
    def train(self, **kwargs):
        """Train function for inherit class.

        Must be written by inherit class
        """
        pass


    @abc.abstractclassmethod
    def eval(self, **kwargs):
        """Eval function for inherit class.

        Must be written by inherit class
        """
        pass

    
    @abc.abstractclassmethod
    def get_loss_on_batch(self, **kwargs):
        """Return batch loss during training model.

        Must be written by inherit class.
        """
        pass


    @abc.abstractclassmethod
    def get_result_on_batch(self, **kwargs):
        """Return batch output logits during eval model.

        Must be written by inherit class.
        """
        pass


    @abc.abstractclassmethod
    def resume_test_at(self, **kwargs):
        """Resume checkpoint and do test.

        Must be written by inherit class.
        """
        pass


    @abc.abstractclassmethod
    def load_examples_features(self, **kwargs) -> tuple:
        """Load examples, features and dataset.
        
        Must be written by inherit class.
        """
        pass

    
    @abc.abstractclassmethod
    def read_examples(self, **kwargs) -> list:
        """Read data from a data file and generate list of InputExamples.

        Must be written by inherit class.
        """
        pass


    @abc.abstractclassmethod
    def convert_examples_to_features(self, **kwargs) -> list:
        """Process the InputExamples into InputFeatures that can be fed into the model.

        Must be written by inherit class.
        """
        pass
