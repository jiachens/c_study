#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Jiachen Sun
@Contact: jiachens@umich.edu
@File: main.py
"""


from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, StepLR, MultiStepLR, ReduceLROnPlateau
from data import PCData_SSL, PCData, PCData_Jigsaw
from model_finetune import PointNet_Rotation, DGCNN_Rotation, PointNet_Jigsaw, PointNet, DGCNN, PointNet_Simple, Pct, DeepSym
import numpy as np
from torch.utils.data import DataLoader
import sys
sys.path.append("./emd/")
import emd_module
from util import cal_loss, IOStream, cross_entropy_with_probs,trades_loss
import sklearn.metrics as metrics
import attack
import time
import model_combine
# EPS=0.05
# ALPHA=0.01
# TRAIN_ITER=7
# TEST_ITER=7

def _init_():
    if not os.path.exists(args.pre_path +'finetune_checkpoints'):
        os.makedirs(args.pre_path +'finetune_checkpoints')
    if not os.path.exists(args.pre_path +'finetune_checkpoints/'+args.exp_name):
        os.makedirs(args.pre_path +'finetune_checkpoints/'+args.exp_name)
    if not os.path.exists(args.pre_path +'finetune_checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs(args.pre_path +'finetune_checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp finetune_main.py '+args.pre_path+'finetune_checkpoints'+'/'+args.exp_name+'/'+'finetune_main.py.backup')
    os.system('cp model_finetune.py '+args.pre_path+'finetune_checkpoints' + '/' + args.exp_name + '/' + 'model_finetune.py.backup')
    os.system('cp util.py '+args.pre_path+'finetune_checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py '+args.pre_path+'finetune_checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')
    os.system('cp attack.py '+args.pre_path+'finetune_checkpoints' + '/' + args.exp_name + '/' + 'attack.py.backup')
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
      m.eval()

def load_pretrain(model, pretrain):
    state_dict = torch.load(pretrain, map_location='cpu')
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for key, val in state_dict.items():
        # print(key)
        if key[:6] == 'module':
            name = key[7:]  # remove 'module.'
        else:
            name = key
        if name[:7] == 'encoder':
            new_state_dict[name[8:]] = val
    model_dict =  model.state_dict()
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    print(f"Load model from {pretrain}")
    return model   

def train(args, io):

    # flag_translate = (args.model == 'pct')
    train_loader = DataLoader(PCData(name=args.dataset, partition='train', num_points=args.num_points), num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(PCData(name=args.dataset, partition='test', num_points=args.num_points), num_workers=8,
                             batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    if args.dataset == 'modelnet40':
        output_channel = 40
    elif args.dataset == 'modelnet10':
        output_channel = 10
    elif args.dataset == 'scanobjectnn':
        output_channel = 15
    elif args.dataset == 'shapenet':
        output_channel = 57

    #Try to load models
    if args.model == 'pointnet':
        model = PointNet(args,output_channels=output_channel).to(device)
    elif args.model == 'dgcnn' and args.combine:
        model = model_combine.DGCNN(args,output_channels=output_channel).to(device)
    elif args.model == 'dgcnn' and  not args.combine:
        model = DGCNN(args,output_channels=output_channel).to(device)
    elif args.model == 'pointnet_simple' and args.combine:
        model =  model_combine.PointNet_Simple(args,output_channels=output_channel).to(device)
    elif args.model == 'pointnet_simple' and  not args.combine:
        model =  PointNet_Simple(args,output_channels=output_channel).to(device)
    elif args.model == 'pct':
        model = Pct(args,output_channels=output_channel).to(device)
    elif args.model == 'deepsym':
        model = DeepSym(args).to(device)
            #saved_model.load_state_dict(torch.load(args.p))
    else:
        raise Exception("Not implemented")

    if args.p != '':  
        if not args.autoencoder:
            saved_model = torch.load(args.p)
            model_dict =  model.state_dict()
            if args.jigsaw:
                state_dict = {k[7:]:v for k,v in saved_model.items() if (k[7:] in model_dict.keys() and not k[7:].startswith('stn') and not k[7:].startswith('fstn'))} # module.
            else:
                state_dict = {k[7:]:v for k,v in saved_model.items() if (k[7:] in model_dict.keys())} # module.
            model_dict.update(state_dict)
            model.load_state_dict(model_dict)
        else:
            model = load_pretrain(model,args.p)
    else:
        if args.model == 'pointnet_simple':
            for name,m in model.named_modules():
                if isinstance(m, (nn.Conv1d, nn.Linear)):
                    nn.init.xavier_uniform_(m.weight)
                    # nn.init.constant_(m.bias, 0.)


    print(str(model))

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr)

    
    if args.scheduler == 'default':
        scheduler = StepLR(opt, 20, 0.7)
    elif args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=0.00001)
    elif args.scheduler == 'piecewise':
        scheduler = MultiStepLR(opt, milestones=[100,150,200], gamma=0.1)
    elif args.scheduler == 'piecewise_1':
        scheduler = MultiStepLR(opt, milestones=[40,80,120], gamma=0.1)
    elif args.scheduler == 'pct':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)

    criterion = cal_loss


    best_test_acc_adv = 0
    best_model = None
    best_epoch = 0


    for epoch in range(args.epochs):
        ####################
        # Train
        ####################

        if args.model == 'pointnet':
            if epoch % 20 == 0:
                for layer in model.modules():
                    if type(layer) == torch.nn.modules.batchnorm.BatchNorm1d:
                        layer.momentum = max(0.01, 0.5**((epoch // 20) + 1))
                    # if type(layer) == model.STN3d or type(layer) == model.STNkd:
                    #     for sublayer in layer:
                    #         if type(sublayer) == torch.nn.modules.batchnorm.BatchNorm1d:
                    #             sublayer.momentum = np.max(0.01, 0.5**((epoch // 20) + 1))

        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []

        # test(args,io,model=model, dataloader = test_loader)

        for data, label, _, _ in train_loader:
            # print(rotated_data.shape)
            # print(rotation_label.shape)
            data, label = data.to(device).float(), label.to(device).long().squeeze()
            batch_size, N, C = data.size()
            data = data.permute(0, 2, 1)

            if args.adversarial:
                data = attack.pgd_attack(model,data,label,eps=args.eps,alpha=args.alpha,iters=args.train_iter,mixup=False) 
                model.train()
            opt.zero_grad()
            logits,trans,trans_feat = model(data)
            loss = criterion(logits, trans_feat, label)
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)

        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                     train_loss*1.0/count,
                                                                                     metrics.accuracy_score(
                                                                                         train_true, train_pred),
                                                                                     metrics.balanced_accuracy_score(
                                                                                         train_true, train_pred))

        io.cprint(outstr)
        if args.scheduler == 'plateau':
            scheduler.step(train_loss*1.0/count)
        else:
            scheduler.step()
        
        acc = test(args,io,model=model, dataloader = test_loader)

        if epoch % 10 == 0 or epoch == args.epochs-1:
            if epoch == args.epochs-1:
                args.test_iter = 200
                args.alpha = 0.005
                io.cprint('Best epoch: %d' % best_epoch)
                io.cprint('Best model in 7-step test:')
                adversarial(args,io,model=best_model, dataloader = test_loader)
                torch.save(best_model.state_dict(), args.pre_path+'finetune_checkpoints/%s/models/model_best.t7' % (args.exp_name))

            acc_adv = adversarial(args,io,model=model, dataloader = test_loader)
            if acc_adv > best_test_acc_adv:
                best_model = model
                best_epoch = epoch
                best_test_acc_adv = acc_adv

            torch.save(model.state_dict(), args.pre_path+'finetune_checkpoints/%s/models/model_epoch%d.t7' % (args.exp_name,epoch))
    return model

def test(args, io,model=None, dataloader=None):

    if dataloader == None:
        test_loader = DataLoader(PCData(name=args.dataset,partition='test', num_points=args.num_points), num_workers=8,
                             batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    else:
        test_loader = dataloader

    device = torch.device("cuda" if args.cuda else "cpu")

    if args.dataset == 'modelnet40':
        output_channel = 40
    elif args.dataset == 'modelnet10':
        output_channel = 10
    elif args.dataset == 'scanobjectnn':
        output_channel = 15
    elif args.dataset == 'shapenet':
        output_channel = 57
    #Try to load models
    if model is None:
        if args.model == 'pointnet':
            model = PointNet(args,output_channels=output_channel).to(device)
        elif args.model == 'dgcnn':
            model = DGCNN(args,output_channels=output_channel).to(device)
        elif args.model == 'pointnet_simple':
            model = PointNet_Simple(args,output_channels=output_channel).to(device)
        elif args.model == 'pct':
            model = Pct(args,output_channels=output_channel).to(device)
        elif args.model == 'deepsym':
            model = DeepSym(args).to(device)
                #saved_model.load_state_dict(torch.load(args.p))
        else:
            raise Exception("Not implemented")
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(args.model_path))

    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    for data, label,_,_ in test_loader:

        data, label = data.to(device).float(), label.to(device).long().squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        # print(data.shape)
        logits,trans,trans_feat = model(data)
        # if args.jigsaw:
        #     logits = logits.view(-1,args.k1**3)
        #     label = label.view(-1,1)[:,0]

        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    io.cprint(outstr)

    return test_acc


def adversarial(args,io,model=None, dataloader=None):

    if dataloader == None:
        test_loader = DataLoader(PCData(name=args.dataset,partition='test', num_points=args.num_points), num_workers=8,
                             batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    else:
        test_loader = dataloader

    device = torch.device("cuda" if args.cuda else "cpu")

    if args.dataset == 'modelnet40':
        output_channel = 40
    elif args.dataset == 'modelnet10':
        output_channel = 10
    elif args.dataset == 'scanobjectnn':
        output_channel = 15
    elif args.dataset == 'shapenet':
        output_channel = 57
    #Try to load models
    if model is None:
        if args.model == 'pointnet':
            model = PointNet(args,output_channels=output_channel).to(device)
        elif args.model == 'dgcnn':
            model = DGCNN(args,output_channels=output_channel).to(device)
        elif args.model == 'pointnet_simple':
            model = PointNet_Simple(args,output_channels=output_channel).to(device)
        elif args.model == 'pct':
            model = Pct(args,output_channels=output_channel).to(device)
        elif args.model == 'deepsym':
            model = DeepSym(args).to(device)
        else:
            raise Exception("Not implemented")
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(args.model_path))

    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    for data, label,_,_ in test_loader:
        data, label = data.to(device).float(), label.to(device).long().squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        adv_data = attack.pgd_attack(model,data,label,eps=args.eps,alpha=args.alpha,iters=args.test_iter,repeat=1,mixup=False)
        logits,trans,trans_feat = model(adv_data)
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Adversarial :: ADV_test acc: %.6f, ADV_test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    io.cprint(outstr)
    return test_acc

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn', 'pointnet_simple', 'pct', 'deepsym'],
                        help='Model to use, [pointnet, dgcnn pointnet_simple]')
    parser.add_argument('--pre_path', type=str, default='./', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 0)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--p', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--eps',type=float,default=0.05,
                        help="Maximum allowed L_inf Perturbation for training")
    parser.add_argument('--alpha',type=float,default=0.01,
                        help="Adversarial training perturbation step size")
    parser.add_argument('--train_iter',type=int,default=7,
                        help="Number of steps taken to create adversarial training inputs")
    parser.add_argument('--test_iter',type=int,default=7,
                        help="Number of steps taken to create adversarial test inputs")
    parser.add_argument('--adversarial',type=bool,default=False,
                        help="Whether to use adversarial examples")
    parser.add_argument('--gpu',type=str,default='0',
                        help="Which gpu to use")
    parser.add_argument('--rotation',type=bool,default=False,
                        help="Whether to use rotation")
    parser.add_argument('--combine',type=bool,default=False,
                        help="Whether to use combine")
    parser.add_argument('--jigsaw',type=bool,default=False,
                        help="Whether to use jigsaw")
    parser.add_argument('--autoencoder',type=bool,default=False,
                        help="Whether to use autoencoder")
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--scheduler',type=str,default='default',
                        help="Which lr scheduler to use")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    _init_()
    print(args.adversarial)
    io = IOStream(args.pre_path+'finetune_checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')
    model = None
    if not args.eval:
        start = time.time()
        # EPS=args.eps
        # ALPHA=args.alpha
        # TRAIN_ITER=args.train_iter
        model=train(args,io)
        end = time.time()
        io.cprint("Training took %.6f hours" % ((end - start)/3600))
    else:
        # EPS=args.eps
        # ALPHA=args.alpha
        # TEST_ITER=args.test_iter
        adversarial(args,io,model=model)
    # start = time.time()
    # if args.model != 'set_transformer': 
    #     saliency_map(args,io,model=model)
    # test(args, io,model=model)
    # TEST_ITER=args.test_iter
    # for eps in [0.025,0.05,0.075,0.1]:
    #     print("EPS:",eps)
    #     EPS=eps
    #     ALPHA=eps/10
    #     adversarial(args,io,model=model)
    # end = time.time()
    # io.cprint("Evaluation took %.6f hours" % ((end - start)/3600))
