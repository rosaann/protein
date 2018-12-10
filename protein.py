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
from tools.model_gen import create_model_on_vgg, create_model_vgg_sim_z,create_model_vgg_sim_z_d7, create_model_mul_line, create_model_resnet_18
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
from tools.visualize_utils import *
import os
import pandas as pd
import cv2
import visdom
from xgboost_model import xgboost_train, test_xg_model
from torchvision import transforms

#from torchsample.regularizers import L1Regularizer

class Protein(object):
    def __init__(self, ifTrain = True, xgb_test_result = None):
        seed = 10
        torch.manual_seed(seed)#为CPU设置随机种子
    
        self.config = Config()
        self.ifTrain = ifTrain
        self.preproc = Data_Preproc()
        self.train_class = 0
        self.xgb_test_result = xgb_test_result
        train_data = xgboost_train(False)
        if self.ifTrain:
            dataset = ProteinDataSet(self.preproc,csv_path='../sample_arg.csv', src_data_list = train_data, start_idx=15)
            self.train_loader = data.DataLoader(dataset, self.config.v('batch_size'), num_workers= 8,
                                               shuffle=True, pin_memory=True)
            
        self.model = create_model_vgg_sim_z()
     #   regularizers = [L1Regularizer(scale=1e-4, module_filter='*line*')]
     #   self.model.set_regularizers(regularizers)
      #  self.model = create_model_resnet_18()
      #  self.model = create_model_mul_line()
      #  self.model = create_model_vgg_sim_z_d7()
        
        self.use_gpu = torch.cuda.is_available()
        #self.use_gpu = False
        if self.use_gpu:
            self.model.cuda()
            torch.cuda.manual_seed(seed)
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
          #  if epoch % int( self.config.v('save_per')) == 0:
            self.save_checkpoints(epoch)
    def test_model(self):
        previous = self.find_previous()
        if previous:
            for epoch, resume_checkpoint in zip(previous[0], previous[1]):
                sys.stdout.write('\rEpoch {epoch:d}/{max_epochs:d}:\n'.format(epoch=epoch, max_epochs=self.config.v('epoches')))
                self.resume_checkpoint(resume_checkpoint)
                   
                self.test_epoch()
                break
           #     self.visualize_epoch(self.model, self.visualize_loader, self.priorbox, self.writer, epoch,  self.use_gpu)
           
    def get_testimg_merge_list_old(self,test_image_dir):
        file_list = []
        for root, dirs, files in os.walk(test_image_dir):
             for file in files:
                if os.path.splitext(file)[1] == '.png':
                    name_part = file.split('_')[0]
                    if_enrolled = False
                    for name_enrolled in file_list:
                        if name_enrolled.find(name_part) >= 0:
                            if_enrolled = True
                            break
                    if if_enrolled == False:
                        file_list.append(name_part)
                        
        
        return file_list 
    def get_testimg_merge_list(self,test_image_dir):
        file_list = []
        df=pd.read_csv('../sample_submission.csv')
        for i, row in df.iterrows():
            file_list.append(row['Id'])
            
        return file_list
    def get_merge_image(self, pre_dir):
        img_name_tails = [ 'red', 'green', 'blue', 'yellow']
        imgs = []
        for tail in img_name_tails:
            img_path = pre_dir+ '_' + tail + '.png'
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE )
            imgs.append(img)
        img_merg = cv2.merge(imgs)
        
        if self.preproc is not None:
            img_merg = self.preproc(img_merg) 
        return img_merg
    
    def get_gray_image(self, pre_dir):
        img_path = pre_dir+ '_' + 'green' + '.png'
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE )
        if self.preproc is not None:
            img = self.preproc(img) 
            
        
        return img
    def test_epoch(self):
        
        self.model.train()
        test_image_dir = os.path.join('../', 'test/')

       # vis = visdom.Visdom(server="http://localhost", port=8888)
        check_i = 0;
        _t = Timer()
        df = pd.DataFrame(columns = ["Id", "Predicted"])
        self.idx_df = 0
        test_image_merge_list = self.get_testimg_merge_list(test_image_dir)
        
        banch_num = int(self.config.v('batch_size'))
        img_list = []
        name_list = []
        print('len ', len(test_image_merge_list))
        
        for i, img_name in enumerate( test_image_merge_list):
            img = self.get_gray_image(test_image_dir + img_name)
          #  
          #  img = Variable( img, volatile=True)
            
            if self.use_gpu:
                img = Variable(img.cuda())
            img_to_add = img
          #      print('img shape ', img.shape)
             #   img_to_add = img.unsqueeze(0)
             #   print('img_to_add shape ', img_to_add.shape)

               # img_to_add.transpose(0, 2, 1, 3)
            img_to_add = img
            if i %  banch_num > 0 and i <= (len(test_image_merge_list) - 1):
                img_list.append(img_to_add)
                name_list.append(img_name)
                if i < (len(test_image_merge_list) - 1):
                   continue
            if i % banch_num == 0:
                if i == 0:
                    img_list.append(img_to_add)
                    name_list.append(img_name)

                    continue
     #       images = images.unsqueeze(0)
            

            _t.tic()
            
        #    print('img_list shape pre 1 ', img_list.shape)
            
           # img_list = transform( img_list)
          #  print('img_list shape pre 2 ', img_list.shape)
            
            img_list = torch.cat(img_list, 0)
            print('img_list shape ', img_list.shape)
            
            if check_i == 3:
                vis.images(img_list[0], win=2, opts={'title': 'Reals'})
                self.visTest(self.model, img_list[0], self.priorbox, self.writer, 1, self.use_gpu)
          #  print('imglist ', img_list.shape)        
            out = self.model(img_list, phase='train')
         #   print('out ', out) 
            for i_im, imname in enumerate(name_list):
                 df.set_value(self.idx_df,'Id', imname )
                 data = out[i_im]
                 result_all = []
                 for t_i, tar_rat in enumerate( data):
                     if tar_rat >=0.5:
                         result_all.append(self.config.v('check_id_list')[t_i])
                 result_xgb = self.xgb_test_result[self.idx_df]
                 for r_x in result_xgb:
                     result_all.append(r_x)
              #   print('da ', data.float())
                 result = ''
                # cla = data.argmax(0).item()
               #  result = str( self.config.v('check_id_list')[ cla])
                 if len(result_all) > 0:
                     result = str(result_all[0])
                     if len(result_all) > 1:
                         for r in result_all[1: ]:
                             result += ' '
                             result += str(r)
                 

                 df.set_value(self.idx_df, 'Predicted', result)
                 self.idx_df += 1;
            img_list = []     
            img_list.append(img_to_add)
            name_list = []
            name_list.append(img_name)

         #   check_i += 1  
        df.to_csv('pred.csv', index=None)
        df.head(10)    
        print('Evaluating detections')
        
        
    def find_previous(self):
        if not os.path.exists(os.path.join(self.config.v('out_dir'), 'checkpoint_list.txt')):
            return False
        with open(os.path.join(self.config.v('out_dir'), 'checkpoint_list.txt'), 'r') as f:
            lineList = f.readlines()
        epoches, resume_checkpoints = [list() for _ in range(2)]
        for line in lineList:
          #  epoch = int(line[line.find('epoch ') + len('epoch '): line.find(':')])
          epoch = 69
          checkpoint = line[line.find(':') + 2:-1]
          epoches.append(epoch)
          resume_checkpoints.append(checkpoint)
        return epoches, resume_checkpoints            
    def resume_checkpoint(self, resume_checkpoint):
        if resume_checkpoint == '' or not os.path.isfile(resume_checkpoint):
            print(("=> no checkpoint found at '{}'".format(resume_checkpoint)))
            return False
        print(("=> loading checkpoint '{:s}'".format(resume_checkpoint)))
        checkpoint = torch.load(resume_checkpoint)

        if 'module.' in list(checkpoint.items())[0][0]:
            pretrained_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
            checkpoint = pretrained_dict

        resume_scope = ''
        # extract the weights based on the resume scope
        if resume_scope != '':
            pretrained_dict = {}
            for k, v in list(checkpoint.items()):
                for resume_key in resume_scope.split(','):
                    if resume_key in k:
                        pretrained_dict[k] = v
                        break
            checkpoint = pretrained_dict

        pretrained_dict = {k: v for k, v in checkpoint.items() if k in self.model.state_dict()}
        # print("=> Resume weigths:")
        # print([k for k, v in list(pretrained_dict.items())])

        checkpoint = self.model.state_dict()

        unresume_dict = set(checkpoint)-set(pretrained_dict)
        if len(unresume_dict) != 0:
            print("=> UNResume weigths:")
            print(unresume_dict)

        checkpoint.update(pretrained_dict)

        return self.model.load_state_dict(checkpoint)

    def train_per_epoch(self, epoch):
        conf_loss = 0
        _t = Timer()
        conf_loss_v = 0
      
        epoch_size = int( len(self.train_loader) )
        
        train_end = int( epoch_size * 0.8);
        batch_iterator = iter(self.train_loader)
        print('epoch_size ', epoch_size, " train_end ", train_end)
        
        
        for iteration  in range(epoch_size):
            images, targets = next(batch_iterator)
          #  print('images ', images.shape)
            if len (images) == 1:
                continue
         #   print('imgs from data_load shape ', images.shape)
            targets = np.array(targets)
           # print('iteration ', iteration)
            if iteration == (train_end - 1):
                if self.use_gpu:
                    images = Variable(images.cuda())
                self.visualize_epoch(images, epoch)
            if iteration <= train_end:
                if self.use_gpu:
                    images = Variable(images.cuda())
                  #  targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
                else:
                    images = Variable(images)
                self.model.train()
                #train:
                _t.tic()
             #   print('img shape ', images.shape)
                out = self.model(images, phase='train')

                self.optimizer.zero_grad()
             #   print('out ', out)
             #   print('targets ', targets.shape)
                loss_c = self.criterion(out, targets)

                # some bugs in coco train2017. maybe the annonation bug.
                if loss_c.data[0] == float("Inf"):
                    continue
                if math.isnan(loss_c.data[0]):
                    continue
             #   if loss_c.data[0] > 100000000:
             #       continue

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
                
                if iteration == (train_end-1):
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
        
            if iteration > train_end:
             #   self.visualize_epoch(model, images[0], targets[0], self.priorbox, writer, epoch, use_gpu)
                #eval:
                if self.use_gpu:
                    images = Variable(images.cuda())
                else:
                    images = Variable(images)
                self.model.eval()
                out = self.model(images, phase='eval')

                # loss
                loss_c = self.criterion(out, targets)
                
                if loss_c.data[0] == float("Inf"):
                    continue
                if math.isnan(loss_c.data[0]):
                    continue
              #  if loss_c.data[0] > 100000000:
              #      continue

                time = _t.toc()

                conf_loss_v += loss_c.data[0]

                # log per iter
                log = '\r==>Eval: || {iters:d}/{epoch_size:d} in {time:.3f}s [{prograss}] ||  cls_loss: {cls_loss:.4f}\r'.format(
                    prograss='#'*int(round(10*iteration/epoch_size)) + '-'*int(round(10*(1-iteration/epoch_size))), iters=iteration, epoch_size=epoch_size,
                    time=time,  cls_loss=loss_c.data[0])
                #print(log)
                sys.stdout.write(log)
                sys.stdout.flush()
           #     self.writer.add_scalar('Eval/conf_loss', conf_loss_v/epoch_size, epoch)
                if iteration == (epoch_size - 1):
                    # eval mAP
             #       prec, rec, ap = cal_pr(label, score, npos)

                    # log per epoch
                    sys.stdout.write('\r')
                    sys.stdout.flush()
                    log = '\r==>Eval: || {iters:d}/{epoch_size:d} in {time:.3f}s [{prograss}] ||  cls_loss: {cls_loss:.4f}\r'.format(
                    prograss='#'*int(round(10*iteration/epoch_size)) + '-'*int(round(10*(1-iteration/epoch_size))), iters=iteration, epoch_size=epoch_size,
                    time=time,  cls_loss=loss_c.data[0])
                    sys.stdout.write(log)
                    sys.stdout.flush()
                    # log for tensorboard
                    self.writer.add_scalar('Eval/conf_loss', conf_loss_v/epoch_size, epoch)
                  #  writer.add_scalar('Eval/mAP', ap, epoch)
                 #   viz_pr_curve(writer, prec, rec, epoch)
                 #   viz_archor_strategy(writer, size, gt_label, epoch)

    def visualize_epoch(self,images, epoch):
        self.model.eval()
     #   for i, image in enumerate(images_list):
        image = Variable( images[0].unsqueeze(0), volatile=True)
        if self.use_gpu:
            image = image.cuda()
    #    print('image shpe', image.shape)
        base_out = viz_module_feature_maps(self.writer, self.model.base, image, module_name='base', epoch=epoch)
    #    extras_out = viz_module_feature_maps(self.writer, self.model.extras, base_out, module_name='extras', epoch=epoch)
        # visualize feature map in feature_extractors
   #     viz_feature_maps(self.writer, self.model(image, 'feature'), module_name='feature_extractors', epoch=epoch)

       # self.model.train()
       # images[0].requires_grad = True
       # images[0].volatile=False
        #base_out = viz_module_grads(writer, model, model.base, images, images, preproc.means, module_name='base', epoch=epoch)
     #   base_out = viz_module_grads(self.writer, self.model, self.model.base, image, image, 0.5, module_name='base', epoch=epoch)

    def save_checkpoints(self, epochs, iters=None):
        if not os.path.exists(self.config.v('out_dir')):
            os.makedirs(self.config.v('out_dir'))
        if iters:
            filename = '_epoch_{:d}_iter_{:d}'.format(epochs, iters) + '.pth'
        else:
            filename = 'epoch_{:d}'.format(epochs) + '.pth'
        filename = os.path.join(self.config.v('out_dir'), filename)
        torch.save(self.model.state_dict(), filename)
        with open(os.path.join(self.config.v('out_dir'), 'checkpoint_list.txt'), 'a') as f:
            f.write('epoch {epoch:d}: {filename}\n'.format(epoch=epochs, filename=filename))
        print('Wrote snapshot to: {:s}'.format(filename))
        
    def trainable_param(self, trainable_scope):
        for param in self.model.parameters():
            param.requires_grad = True

        trainable_param = []
        for module in trainable_scope.split(','):
            if hasattr(self.model, module):
                # print(getattr(self.model, module))
                for param in getattr(self.model, module).parameters():
                    param.requires_grad = True
                trainable_param.extend(getattr(self.model, module).parameters())

        return [self.model.parameters]
        
    def configure_optimizer(self, trainable_param):
     #   optimizer = optim.SGD(self.model.parameters(), lr= self.config.v('learn_rate'),
     #                   momentum=self.config.v('momentum'), weight_decay= self.config.v('weight_decay'))
     #   optimizer = optim.RMSprop(trainable_param, lr=self.config.v('learn_rate'),
     #                   momentum=self.config.v('momentum'), alpha=self.config.v('momentum_2'), eps=cfg.EPS, weight_decay=self.config.v('weight_decay'))
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.v('learn_rate'),
                        betas=(self.config.v('momentum'), self.config.v('momentum_2')), eps=self.config.v('eps'), weight_decay=self.config.v('weight_decay'))
        
        return optimizer


    def configure_lr_scheduler(self, optimizer):
        scheduler = lr_scheduler.StepLR(optimizer, step_size=self.config.v('lr_steps')[0], gamma=self.config.v('lr_gamma'))
       # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=elf.config.v('lr_steps'), gamma=self.config.v('lr_gamma'))
       # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=self.config.v('lr_gamma'))
       # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.v('epoches'))
        return scheduler
        
def train_model():
  #  xgboost_train()
    print('start ')
    s = Protein(ifTrain = True)
    s.train_model()
    return True

def test_model():
    xgb_test = test_xg_model()
    s = Protein(ifTrain = False, xgb_test_result = xgb_test)
    s.test_model()
    return True       
