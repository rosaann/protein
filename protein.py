#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 13:58:12 2018

@author: zl
"""
from tools.protein_dataset import ProteinDataSet
from tools.data_preproc import Data_Preproc
import torch.utils.data as data
from config import Config
from tools.model_gen import create_model_on_vgg
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler

class Protein(object):
    def __init__(self, ifTrain = True):
        self.config = Config()
        self.ifTrain = ifTrain
        if self.ifTrain:
            dataset = ProteinDataSet(Data_Preproc())
            train_loader = data.DataLoader(dataset, self.config.v('batch_size'), num_workers= 8,
                                  shuffle=False, pin_memory=True)
            
        self.model = create_model_on_vgg()
        
        self.use_gpu = torch.cuda.is_available()
        #self.use_gpu = False
        if self.use_gpu:
            self.model.cuda()
            cudnn.benchmark = True
            if torch.cuda.device_count() > 1:
                 self.model = torch.nn.DataParallel(self.model).module
                 
                 
        trainable_param = self.trainable_param('base,extras,norm,loc,conf')
       # print('trainable_param ', trainable_param)
        self.optimizer = self.configure_optimizer(trainable_param)
        self.exp_lr_scheduler = self.configure_lr_scheduler(self.optimizer)
        self.max_epochs = self.config.v('epoches')
    
    def trainable_param(self, trainable_scope):
        for param in self.model.parameters():
            param.requires_grad = False

        trainable_param = []
        for module in trainable_scope.split(','):
            if hasattr(self.model, module):
                # print(getattr(self.model, module))
                for param in getattr(self.model, module).parameters():
                    param.requires_grad = True
                trainable_param.extend(getattr(self.model, module).parameters())

        return trainable_param
        
    def configure_optimizer(self, trainable_param):
        optimizer = optim.SGD(trainable_param, lr= self.config.v('learn_rate'),
                        momentum=self.config.v('momentum'), weight_decay= self.config.v('weight_decay'))
     #   optimizer = optim.RMSprop(trainable_param, lr=self.config.v('learn_rate'),
     #                   momentum=self.config.v('momentum'), alpha=self.config.v('momentum_2'), eps=cfg.EPS, weight_decay=self.config.v('weight_decay'))
     #   optimizer = optim.Adam(trainable_param, lr=self.config.v('learn_rate'),
    #                    betas=(self.config.v('momentum'), self.config.v('momentum_2')), eps=self.config.v('eps'), weight_decay=self.config.v('weight_decay'))
        
        return optimizer


    def configure_lr_scheduler(self, optimizer):
        scheduler = lr_scheduler.StepLR(optimizer, step_size=self.config.v('lr_steps')[0], gamma=self.config.v('lr_gamma'))
       # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=elf.config.v('lr_steps'), gamma=self.config.v('lr_gamma'))
       # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=self.config.v('lr_gamma'))
       # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.v('epoches'))
        return scheduler
        
        