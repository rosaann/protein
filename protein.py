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
import torch
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from models.other_layers.multiclass_loss import MultiClassLoss
from tools.timer import Timer
import math
import sys
from tensorboardX import SummaryWriter
import numpy as np

class Protein(object):
    def __init__(self, ifTrain = True):
        self.config = Config()
        self.ifTrain = ifTrain
        if self.ifTrain:
            dataset = ProteinDataSet(Data_Preproc())
            self.train_loader = data.DataLoader(dataset, self.config.v('batch_size'), num_workers= 8,
                                  shuffle=False, pin_memory=True)
            
        self.model = create_model_on_vgg()
        
        self.use_gpu = torch.cuda.is_available()
        #self.use_gpu = False
        if self.use_gpu:
            self.model.cuda()
            cudnn.benchmark = True
            if torch.cuda.device_count() > 1:
                 self.model = torch.nn.DataParallel(self.model).module
                 
        print('Model architectures:\n{}\n'.format(self.model))         
        trainable_param = self.trainable_param('base,extras,norm,loc,conf')
       # print('trainable_param ', trainable_param)
        self.optimizer = self.configure_optimizer(trainable_param)
        self.exp_lr_scheduler = self.configure_lr_scheduler(self.optimizer)
        self.max_epochs = self.config.v('epoches')
        
        self.criterion = MultiClassLoss( self.use_gpu)
        self.writer = SummaryWriter(self.config.v('out_dir'))
        
    def train_model(self):
        for epoch in range( self.max_epochs):
            self.train_per_epoch(epoch)
    
    def train_per_epoch(self, epoch):
        epoch_size = int( len(self.train_loader) / self.config.v('batch_size'))
        batch_iterator = iter(self.train_loader)
        train_end = int( epoch_size * 0.8);
        conf_loss = 0
        _t = Timer()
        
        conf_loss_v = 0
        
        for iteration in iter(range((epoch_size))):
            images, targets = next(batch_iterator)
            targets = np.array(targets)
          #  if iteration > train_end and iteration < train_end + 10:
          #      self.visualize_epoch(model, int(iteration) * int(self.cfg.DATASET.TRAIN_BATCH_SIZE), self.priorbox, writer, epoch, use_gpu)
            if iteration <= train_end:
                if self.use_gpu:
                    images = Variable(images.cuda())
                  #  targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
                else:
                    images = Variable(images)
                    targets = [Variable(anno, volatile=True) for anno in targets]
                self.model.train()
                #train:
                _t.tic()
                out = self.model(images, phase='train')

                self.optimizer.zero_grad()
                loss_c = self.criterion(out, targets)

                # some bugs in coco train2017. maybe the annonation bug.
                if loss_c.data[0] == float("Inf"):
                    continue
                if math.isnan(loss_c.data[0]):
                    continue
                if loss_c.data[0] > 100:
                    continue

                loss_c.backward()
                self.optimizer.step()

                time = _t.toc()
                conf_loss += loss_c.data[0]

                # log per iter
                log = '\r==>Train: || {iters:d}/{epoch_size:d} in {time:.3f}s [{prograss}] ||  cls_loss: {cls_loss:.4f}\r'.format(
                    prograss='#'*int(round(10*iteration/epoch_size)) + '-'*int(round(10*(1-iteration/epoch_size))), iters=iteration, epoch_size=epoch_size,
                    time=time, cls_loss=loss_c.data[0])

                sys.stdout.write(log)
                sys.stdout.flush()
                
                if iteration == train_end:
                    # log per epoch
                    sys.stdout.write('\r')
                    sys.stdout.flush()
                    lr = self.optimizer.param_groups[0]['lr']
                    log = '\r==>Train: || Total_time: {time:.3f}s ||  conf_loss: {conf_loss:.4f} || lr: {lr:.6f}\n'.format(lr=lr,
                        time=_t.total_time,  conf_loss=conf_loss/epoch_size)
                    sys.stdout.write(log)
                    sys.stdout.flush()
                 #   print(log)
                    # log for tensorboard
                    self.writer.add_scalar('Train/conf_loss', conf_loss/epoch_size, epoch)
                    self.writer.add_scalar('Train/lr', lr, epoch)
                    
                    conf_loss = 0

        train_end = int( epoch_size * 0.9);
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
        
def train_model():
    s = Protein(ifTrain = True)
    s.train_model()
    return True

def test_model():
    s = Protein(ifTrain = False)
    s.test_model()
    return True       