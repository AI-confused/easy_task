# -*- coding: utf-8 -*-
# AUTHOR: Li Yun Liang
# DATE: 21-7-9

import logging
import random
import os
import json
import sys
import numpy as np
from datetime import datetime
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.nn.parallel as para
import tqdm
from .base_utils import *
from .base_result import *


# def set_optimizer_params_grad(named_params_optimizer, named_params_model, test_nan=False):
#     """
#         Utility function for optimize_on_cpu and 16-bits training.
#         Copy the gradient of the GPU parameters to the CPU/RAMM copy of the model
#     """
#     is_nan = False
#     for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
#         if name_opti != name_model:
#             logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
#             raise ValueError
#         if param_model.grad is not None:
#             if test_nan and torch.isnan(param_model.grad).sum() > 0:
#                 is_nan = True
#             if param_opti.grad is None:
#                 param_opti.grad = torch.nn.Parameter(param_opti.data.new().resize_(*param_opti.data.size()))
#             param_opti.grad.data.copy_(param_model.grad.data)
#         else:
#             param_opti.grad = None
#     return is_nan


# def copy_optimizer_params_to_model(named_params_model, named_params_optimizer):
#     """
#         Utility function for optimize_on_cpu and 16-bits training.
#         Copy the parameters optimized on CPU/RAM back to the model on GPU
#     """
#     for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
#         if name_opti != name_model:
#             logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
#             raise ValueError
#         param_model.data.copy_(param_opti.data)


class BasePytorchTask(object):
    def __init__(self, setting):
        """
        Basic task to support deep learning models on Pytorch
        Args:
            setting (class DEETaskSetting): hyperparameters of PytorchTask.
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


    def __check_setting_validity(self):
        """
        check task setting parameters are valid or not.
        Args:
            /
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
        """
        init device
        Args:
            /
        """
        self.logger.info('='*20 + 'Init Device' + '='*20)
        
        # set device
        self.device = torch.device("cuda" if torch.cuda.is_available() and not self.setting.no_cuda else "cpu")
        self.n_gpu = torch.cuda.device_count()
        self.logger.info("device {} / n_gpu {}".format(self.device, self.n_gpu))


    def _load_data(self, load_example_func, convert_feature_func, load_train, load_dev, load_test, **kwargs):
        """
        Load dataset and construct model's examples, features and dataset.
        Args:
            load_example_func(func): read_examples
            convert_feature_func(func): convert_examples_to_features
            load_train(bool): load train portion or not
            load_dev(bool): load dev portion or not
            load_test(bool): load test portion or not
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
   

    def load_example_feature_dataset(self, load_example_func, convert_to_feature_func, convert_to_dataset_func,
                                     file_name=None, file_path=None, only_inference=False):
        if file_name is None and file_path is None:
            raise Exception('Either file name or file path should be provided')

        if file_path is None:
            file_path = os.path.join(self.setting.data_dir, file_name)

        if os.path.exists(file_path):
            self.logger.info('Load example feature dataset from {}'.format(file_path))
            examples = load_example_func(file_path, self.setting.percent, only_inference)
            features = convert_to_feature_func(examples)
            dataset = convert_to_dataset_func(features)
        else:
            self.logger.info('Warning: file does not exists, {}'.format(file_path))
            examples = None
            features = None
            dataset = None

        return examples, features, dataset


    def _decorate_model(self):
        self.logger.info('='*20 + 'Decorate Model' + '='*20)

        # self.model.to(self.device)
        # self.logger.info('Set model device to {}'.format(str(self.device)))
        
        if self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
            self.logger.info('Multi-gpu training')
        elif self.n_gpu == 1:
            self.logger.info('Single-gpu training')
        else:
            self.logger.info('no cuda training')
        
        self.model.to(self.device)
        self.logger.info('Set model device to {}'.format(str(self.device)))


    def get_latest_cpt_epoch(self):
        """
        Continue training with the latest epoch from the saved model directory
        Args:
            /
        """
        prev_epochs = []
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
        

    def reset_random_seed(self, seed=None):
        """
        reset random seed during task.
        Args:
            seed(int): random seed.
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

    # def in_multi_cuda_mode(self):
    #     return self.n_gpu >= 0


    # def _init_summary_writer(self):
    #     """
    #     init tensorboard summary writer.
    #     """
    #     if self.is_master_node():
    #         self.logger.info('Init Summary Writer')
    #         current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    #         sum_dir = '{}-{}'.format(self.setting.summary_dir_name, current_time)
    #         # self.summary_writer = SummaryWriter(sum_dir)
    #         self.logger.info('Writing summary into {}'.format(sum_dir))

    #     if self.in_distributed_mode():
    #         # TODO: maybe this can be removed
    #         dist.barrier()

    

    # def _init_bert_optimizer(self):
    #     self.logger.info('='*20 + 'Init Bert Optimizer' + '='*20)
    #     self.optimizer, self.num_train_steps, self.model_named_parameters = \
    #         self.reset_bert_optimizer()

    # def reset_bert_optimizer(self):
    #     # Prepare optimizer
    #     if self.setting.fp16:
    #         model_named_parameters = [(n, param.clone().detach().to('cpu').float().requires_grad_())
    #                                   for n, param in self.model.named_parameters()]
    #     elif self.setting.optimize_on_cpu:
    #         model_named_parameters = [(n, param.clone().detach().to('cpu').requires_grad_())
    #                                   for n, param in self.model.named_parameters()]
    #     else:
    #         model_named_parameters = list(self.model.named_parameters())

    #     no_decay = ['bias', 'gamma', 'beta']
    #     optimizer_grouped_parameters = [
    #         {
    #             'params': [p for n, p in model_named_parameters if n not in no_decay],
    #             'weight_decay_rate': 0.01
    #         },
    #         {
    #             'params': [p for n, p in model_named_parameters if n in no_decay],
    #             'weight_decay_rate': 0.0
    #         }
    #     ]

    #     num_train_steps = int(len(self.train_examples)
    #                           / self.setting.train_batch_size
    #                           / self.setting.gradient_accumulation_steps
    #                           * self.setting.num_train_epochs)

    #     optimizer = BertAdam(optimizer_grouped_parameters,
    #                          lr=self.setting.learning_rate,
    #                          warmup=self.setting.warmup_proportion,
    #                          t_total=num_train_steps)

    #     return optimizer, num_train_steps, model_named_parameters

    def prepare_data_loader(self, dataset, batch_size, rand_flag=True, collate_fn=None):
        """
        prepare data loader

        Args:
            dataset (list(DEEFeature)): 
            batch_size (int): 
            rand_flag (bool): RandomSampler
        """
        if rand_flag:
            data_sampler = RandomSampler(dataset)
        else:
            data_sampler = SequentialSampler(dataset)

        if collate_fn is not None: # Retro
            dataloader = DataLoader(dataset,
                                    batch_size=batch_size,
                                    sampler=data_sampler,
                                    collate_fn=collate_fn)
        else:# D2E
            dataloader = DataLoader(dataset,
                                    batch_size=batch_size,
                                    sampler=data_sampler,
                                    collate_fn=self.custom_collate_fn)

        return dataloader

    # def prepare_dist_data_loader(self, dataset, batch_size, epoch=0):
    #     # prepare distributed data loader
    #     data_sampler = DistributedSampler(dataset)
    #     data_sampler.set_epoch(epoch)

    #     if self.custom_collate_fn is None:
    #         dataloader = DataLoader(dataset,
    #                                 batch_size=batch_size,
    #                                 sampler=data_sampler)
    #     else:
    #         dataloader = DataLoader(dataset,
    #                                 batch_size=batch_size,
    #                                 sampler=data_sampler,
    #                                 collate_fn=self.custom_collate_fn)
    #     return dataloader

    # def get_current_train_batch_size(self):
    #     if self.in_distributed_mode():
    #         train_batch_size = max(self.setting.train_batch_size // dist.get_world_size(), 1)
    #     else:
    #         train_batch_size = self.setting.train_batch_size

    #     return train_batch_size

    def resume_save_eval_at(self, epoch, batch_size, resume_cpt_flag=False, save_cpt_flag=True):
        if self.is_master_node() and epoch >= 0:
            self.logger.info('\nPROGRESS: {}\n'.format(epoch / self.setting.num_train_epochs))

        if resume_cpt_flag:
            self.resume_checkpoint(cpt_file_name='{}.cpt.{}'.format(self.setting.cpt_file_name, epoch),
                               resume_model=True, resume_optimizer=True)

        if self.is_master_node() and save_cpt_flag:
            self.save_checkpoint(cpt_file_name='{}.cpt.{}'.format(self.setting.cpt_file_name, epoch), epoch=epoch)

    def set_batch_to_device(self, batch):
        """
        put batch data into cuda.
        Args:
            batch (list):
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
        """
        base task train func with a set of parameters.
        Args:
            /
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
        train_loss = 0
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
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if self.setting.gradient_accumulation_steps > 1:
                    loss = loss / self.setting.gradient_accumulation_steps

                # backward
                loss.backward()

                loss_scalar = loss.item()
                tr_loss += loss_scalar
                train_loss = round(tr_loss * self.setting.gradient_accumulation_steps / (nb_tr_steps+1), 4)
                bar.set_description('loss {}'.format(train_loss))
                # self.summary_writer.add_scalar('Loss', loss_scalar, global_step=global_step)
                nb_tr_examples += self.setting.train_batch_size
                nb_tr_steps += 1
                if (step + 1) % self.setting.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.model.zero_grad()
                    global_step += 1

            self.eval(epoch_idx + 1)


    def base_eval(self, epoch, data_type, eval_examples, eval_features, eval_dataset, **kwargs):
        """
        Args: 
            epoch(int): eval epoch
            data_type(str): 'dev' or 'test'
            /
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


    def save_checkpoint(self, cpt_file_name=None, epoch=None):
        """
        save save_checkpoint file at model path.

        Args:
            cpt_file_name (str): saved file name.
            epoch (int): num of training epoch.
        """
        self.logger.info('='*20 + 'Dump Checkpoint' + '='*20)
        if cpt_file_name is None:
            cpt_file_name = self.setting.cpt_file_name
        cpt_file_path = os.path.join(self.setting.model_dir, cpt_file_name)
        self.logger.info('Dump checkpoint into {}'.format(cpt_file_path))

        store_dict = {
            'setting': self.setting.__dict__,
        }

        if self.model:
            if isinstance(self.model, para.DataParallel) or \
                    isinstance(self.model, para.DistributedDataParallel):
                model_state = self.model.module.state_dict()
            else:
                model_state = self.model.state_dict()
            store_dict['model_state'] = model_state
        else:
            self.logger.info('No model state is dumped', level=logging.WARNING)

        if self.optimizer:
            store_dict['optimizer_state'] = self.optimizer.state_dict()
        else:
            self.logger.info('No optimizer state is dumped', level=logging.WARNING)

        if epoch:
            store_dict['epoch'] = epoch

        torch.save(store_dict, cpt_file_path)


    def resume_checkpoint(self, cpt_file_name=None, resume_model=True, resume_optimizer=False, strict=False):
        """
        load checkpoint from saved file.
        Args:
            cpt_file_path (str): saved model path.
            cpt_file_name (str): saved model file name.
            resume_model (bool): load model weights.
            resume_optimizer (bool): load optimizer weights.
            strict (bool): /
        Example: /
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


# def average_gradients(model):
#     """ Gradient averaging. """
#     size = float(dist.get_world_size())
#     for name, param in model.named_parameters():
#         try:
#             dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
#             param.grad.data /= size
#         except Exception as e:
#             logger.error('Error when all_reduce parameter {}, size={}, grad_type={}, error message {}'.format(
#                 name, param.size(), param.grad.data.dtype, repr(e)
#             ))









