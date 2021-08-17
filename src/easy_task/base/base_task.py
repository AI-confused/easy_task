"""
-*- coding: utf-8 -*-
@author: black_tears
@time: 2021-07-09
@description: base module of task definition.
"""


import logging
import random
import os
import sys
import json
import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.nn.parallel as para
from src.easy_task.base.base_setting import TaskSetting
from src.easy_task.base.base_utils import BaseUtils


class BasePytorchTask(object):
    def __init__(self, setting: TaskSetting):
        """Basic task to support deep learning models on Pytorch.

        Custom pytorch task should inherit this class and input task_setting.

        @setting: hyperparameters of Pytorch Task.
        """
        self.setting = setting
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # general initialization
        self.__check_setting_validity()
        self.__init_device()
        self.reset_random_seed()

        # task-specific initialization
        self.custom_collate_fn = None
        self.train_examples = None
        self.train_features = None
        self.train_dataset = None
        self.dev_examples = None
        self.dev_features = None
        self.dev_dataset = None
        self.test_examples = None
        self.test_features = None
        self.test_dataset = None
        self.model = None
        self.optimizer = None
        self.custom_collate_fn_train = None
        self.custom_collate_fn_eval = None
        self.num_train_steps = None
        self.model_named_parameters = None
        self.best_dev_score = None
        self.best_test_score = None
        self.output_result = None


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


    def load_data(self, load_example_func, convert_feature_func, load_train: bool, load_dev: bool, load_test: bool, **kwargs):
        """Load dataset and construct model's examples, features and dataset.

        @load_example_func(func): read_examples
        @convert_feature_func(func): convert_examples_to_features
        @load_train: load train portion or not
        @load_dev: load dev portion or not
        @load_test: load test portion or not
        """
        self.logger.info('='*20 + 'load dataset' + '='*20)

        #load train portion
        if load_train:
            self.logger.info('Load train portion')
            self.train_examples, self.train_features, self.train_dataset = self.load_examples_features(load_example_func, convert_feature_func, 'train', self.setting.train_file_name, 1)
            self.logger.info('Load train portion done')
            self.logger.info('training examples: {}, features: {} at max_sequence_len: {}'.format(len(self.train_examples), len(self.train_features), self.setting.max_seq_len))
        else:
            self.logger.info('Do not load train portion')

        # load dev portion
        if load_dev:
            self.logger.info('Load dev portion')
            self.dev_examples, self.dev_features, self.dev_dataset = self.load_examples_features(load_example_func, convert_feature_func, 'dev', self.setting.dev_file_name, 0)
            self.logger.info('Load dev portion done!')
            self.logger.info('dev examples: {}, features: {} at max_sequence_len: {}'.format(len(self.dev_examples), len(self.dev_features), self.setting.max_seq_len))
        else:
            self.logger.info('Do not load dev portion')

        # load test portion
        if load_test:
            self.logger.info('Load test portion')
            self.test_examples, self.test_features, self.test_dataset = self.load_examples_features(load_example_func, convert_feature_func, 'test', self.setting.test_file_name, 0)
            self.logger.info('Load test portion done!')
            self.logger.info('test examples: {}, features: {} at max_sequence_len: {}'.format(len(self.test_examples), len(self.test_features), self.setting.max_seq_len))
        else:
            self.logger.info('Do not load test portion')


    def decorate_model(self):
        """Put model on device.
        """
        self.logger.info('='*20 + 'Decorate Model' + '='*20)
        
        if self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
            self.logger.info('Multi-gpu training')
        elif self.n_gpu == 1:
            self.logger.info('Single-gpu training')
        else:
            self.logger.info('no cuda training')
        
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
                    epoch = int(fn.split('.')[-1])
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


    def prepare_data_loader(self, dataset: list, batch_size: int, rand_flag: bool=True, collate_fn=None):
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
                                    sampler=data_sampler,
                                    collate_fn=self.custom_collate_fn)

        return dataloader


    def set_batch_to_device(self, batch: list):
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


    def base_train(self, **kwargs):
        """Base task train func with a set of parameters.
        
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

        # prepare data loader
        train_dataloader = self.prepare_data_loader(
            self.train_dataset, self.setting.train_batch_size, rand_flag=True, collate_fn=self.custom_collate_fn_train
        )        

        # start training
        global_step = 0
        self.train_loss = 0
        self.logger.info('start training~')
        for epoch_idx in tqdm.trange(kwargs['base_epoch_idx'], int(self.setting.num_train_epochs), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            bar = tqdm.tqdm(train_dataloader)
            for step, batch in enumerate(bar):
                batch = self.set_batch_to_device(batch)
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

            self.eval(epoch_idx + 1)


    def base_eval(self, epoch: int, data_type: str, eval_examples: list, eval_features: list, eval_dataset: list, **kwargs):
        """Base task eval func with a set of parameters.

        @epoch: eval epoch
        @data_type: 'dev' or 'test'
        @eval_examples: list of InputExample
        @eval_features: list of InputFeature
        @eval_dataset: list of InputFeature
        """
        assert self.model is not None
        self.logger.info('=' * 20 + 'Start Evaluation/{}'.format(data_type) + '=' * 20)
        self.logger.info('\nPROGRESS: {}\n'.format(epoch / self.setting.num_train_epochs))
        self.logger.info("\tNum examples = {}".format(len(eval_examples)))
        self.logger.info("\tNum features = {}".format(len(eval_features)))
        self.logger.info("\tBatch size = {}".format(self.setting.eval_batch_size))

        # prepare data loader
        eval_dataloader = self.prepare_data_loader(
            eval_dataset, self.setting.eval_batch_size, rand_flag=False, collate_fn=self.custom_collate_fn_eval
        )

        # enter eval mode
        self.model.eval()

        # do eval
        for batch in tqdm.tqdm(eval_dataloader, desc='Iteration'):
            batch = self.set_batch_to_device(batch)

            with torch.no_grad():
                batch_output, batch_label = self.get_result_on_batch(batch)
                self.result.update_batch(batch_outputs=batch_output, batch_labels=batch_label)


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


    def resume_checkpoint(self, cpt_file_name: str=None, resume_model: bool=True, resume_optimizer: bool=False, strict: bool=False):
        """Load checkpoint from saved file.

        @cpt_file_name: saved model file name.
        @resume_model: load model weights.
        @resume_optimizer: load optimizer weights.
        @strict: /
        """
        self.logger.info('='*20 + 'Resume Checkpoint' + '='*20)

        # build checkpoint file path
        if cpt_file_name == None:
            raise ValueError('cpt_file_name should not be None')
        cpt_file_path = os.path.join(self.setting.model_dir, cpt_file_name)
        
        if os.path.exists(cpt_file_path):
            self.logger.info('Resume checkpoint from {}'.format(cpt_file_path))
        else:
            self.logger.info('Checkpoint does not exist, {}'.format(cpt_file_path), level=logging.WARNING)
            return

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


    def write_results(self):
        """Write results to output file.
        """
        result_file = os.path.join(self.setting.output_dir, '{}_result.json'.format(self.output_result['result_type']))
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



