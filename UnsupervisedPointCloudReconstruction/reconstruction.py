#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@File: reconstruction.py
@Time: 2020/1/2 10:26 AM
"""

import os
import sys
import time
import shutil
import torch
import torch.optim as optim
import numpy as np
import attack

from tensorboardX import SummaryWriter

from model import ReconstructionNet
from dataset import Dataset
from utils import Logger


class Reconstruction(object):
    def __init__(self, args):
        self.args = args
        self.dataset_name = args.dataset
        if args.epochs != None:
            self.epochs = args.epochs
        elif args.encoder == 'pointnet_simple':
            self.epochs = 250
        elif args.encoder == 'dgcnn':
            self.epochs = 250
        elif args.encoder == 'pct':
            self.epochs = 250
        self.batch_size = args.batch_size
        self.snapshot_interval = args.snapshot_interval
        self.no_cuda = args.no_cuda
        self.model_path = args.model_path

        # create exp directory
        file = [f for f in args.model_path.split('/')]
        if args.exp_name != None:
            self.experiment_id = "Reconstruct_" + args.exp_name
        elif file[-2] == 'models':
            self.experiment_id = file[-3]
        else:
            self.experiment_id = "Reconstruct" + time.strftime('%m%d%H%M%S')
        snapshot_root = args.pre_path + 'snapshot/%s' % self.experiment_id
        tensorboard_root = args.pre_path + 'tensorboard/%s' % self.experiment_id
        self.save_dir = os.path.join(snapshot_root, 'models/')
        self.tboard_dir = tensorboard_root

        # check arguments
        if self.model_path == '':
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            else:
                choose = "y"
                if choose == "y":
                    shutil.rmtree(self.save_dir)
                    os.makedirs(self.save_dir)
                else:
                    sys.exit(0)
            if not os.path.exists(self.tboard_dir):
                os.makedirs(self.tboard_dir)
            else:
                shutil.rmtree(self.tboard_dir)
                os.makedirs(self.tboard_dir)
        sys.stdout = Logger(os.path.join(snapshot_root, 'log.txt'))
        self.writer = SummaryWriter(log_dir=self.tboard_dir)

        # print args
        print(str(args))

        # get gpu id
        gids = ''.join(args.gpu.split())
        self.gpu_ids = [int(gid) for gid in gids.split(',')]
        self.first_gpu = self.gpu_ids[0]

        # generate dataset
        self.train_dataset = Dataset(
                root=args.dataset_root,
                dataset_name=args.dataset,
                split='train',
                num_points=args.num_points,
                random_translate=args.use_translate,
                random_rotate=True,
                random_jitter=args.use_jitter
            )
        self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.workers
            )
        print("Training set size:", self.train_loader.dataset.__len__())

        # initialize model
        self.model = ReconstructionNet(args)
        if self.model_path != '':
            self._load_pretrain(args.model_path)

        # load model to gpu
        if not self.no_cuda:
            if len(self.gpu_ids) != 1:  # multiple gpus
                self.model = torch.nn.DataParallel(self.model.cuda(self.first_gpu), self.gpu_ids)
            else:
                self.model = self.model.cuda(self.gpu_ids[0])
        
        # initialize optimizer
        self.parameter = self.model.parameters()
        self.optimizer = optim.Adam(self.parameter, lr=0.0001*16/args.batch_size, betas=(0.9, 0.999), weight_decay=1e-6)


    def run(self):
        self.train_hist = {
            'loss': [],
            'per_epoch_time': [],
            'total_time': []
        }
        best_loss = 1000000000
        print('Training start!!')
        start_time = time.time()
        self.model.train()
        if self.model_path != '':
            start_epoch = self.model_path[-7:-4]
            if start_epoch[0] == '_':
                start_epoch = start_epoch[1:]
            start_epoch = int(start_epoch)
        else:
            start_epoch = 0
        for epoch in range(start_epoch, self.epochs):
            loss = self.train_epoch(epoch)
            
            # save snapeshot
            if (epoch + 1) % self.snapshot_interval == 0:
                self._snapshot(epoch + 1)
                if loss < best_loss:
                    best_loss = loss
                    self._snapshot('best')
            
            # save tensorboard
            if self.writer:
                self.writer.add_scalar('Train Loss', self.train_hist['loss'][-1], epoch)
                self.writer.add_scalar('Learning Rate', self._get_lr(), epoch)
        
        # finish all epoch
        self._snapshot(epoch + 1)
        if loss < best_loss:
            best_loss = loss
            self._snapshot('best')
        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.epochs, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")


    def train_epoch(self, epoch):
        epoch_start_time = time.time()
        loss_buf = []
        num_batch = int(len(self.train_loader.dataset) / self.batch_size)
        for iter, (pts, _) in enumerate(self.train_loader):
            if not self.no_cuda:
                pts = pts.cuda(self.first_gpu)

            # forward
            data = pts.permute(0, 2, 1)
            self.optimizer.zero_grad()
            output, feature = self.model(data)
            loss_tv = 0
            if self.args.adversarial:
                adv_data = attack.pgd_attack(self.model,data,feature,eps=0.05,alpha=0.01,iters=7)
                self.model.train()
                output, feature_adv = self.model(adv_data)
                loss_tv = torch.mean(torch.abs(feature-feature_adv))

            # loss
            if len(self.gpu_ids) != 1:  # multiple gpus
                loss = self.model.module.get_loss(pts, output) + loss_tv
            else:
                loss = self.model.get_loss(pts, output) + loss_tv

            # backward
            loss.backward()
            self.optimizer.step()
            loss_buf.append(loss.detach().cpu().numpy())

        # finish one epoch
        epoch_time = time.time() - epoch_start_time
        self.train_hist['per_epoch_time'].append(epoch_time)
        self.train_hist['loss'].append(np.mean(loss_buf))
        print(f'Epoch {epoch+1}: Loss {np.mean(loss_buf)}, time {epoch_time:.4f}s')
        return np.mean(loss_buf)


    def _snapshot(self, epoch):
        state_dict = self.model.state_dict()
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for key, val in state_dict.items():
            if key[:6] == 'module':
                name = key[7:]  # remove 'module.'
            else:
                name = key
            new_state_dict[name] = val
        save_dir = os.path.join(self.save_dir, self.dataset_name)
        torch.save(new_state_dict, save_dir + "_" + str(epoch) + '.pkl')
        print(f"Save model to {save_dir}_{str(epoch)}.pkl")


    def _load_pretrain(self, pretrain):
        state_dict = torch.load(pretrain, map_location='cpu')
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for key, val in state_dict.items():
            if key[:6] == 'module':
                name = key[7:]  # remove 'module.'
            else:
                name = key
            new_state_dict[name] = val
        self.model.load_state_dict(new_state_dict)
        print(f"Load model from {pretrain}")


    def _get_lr(self, group=0):
        return self.optimizer.param_groups[group]['lr']
