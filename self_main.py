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
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, StepLR, MultiStepLR
from data import PCData_SSL, PCData, PCData_Jigsaw
from model_finetune import PointNet_Rotation, DGCNN_Rotation, PointNet_Jigsaw, DGCNN_Jigsaw, DeepSym_Rotation, DeepSym_Jigsaw, Pct_Jigsaw, Pct_Rotation
import numpy as np
from torch.utils.data import DataLoader
import sys
sys.path.append("./emd/")
import emd_module
from util import cal_loss, IOStream, cross_entropy_with_probs,trades_loss
import sklearn.metrics as metrics
import attack
import time

def _init_():
    if not os.path.exists(args.pre_path + 'ssl_checkpoints'):
        os.makedirs(args.pre_path + 'ssl_checkpoints')
    if not os.path.exists(args.pre_path + 'ssl_checkpoints/'+args.exp_name):
        os.makedirs(args.pre_path + 'ssl_checkpoints/'+args.exp_name)
    if not os.path.exists(args.pre_path + 'ssl_checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs(args.pre_path + 'ssl_checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp self_main.py ' + args.pre_path + 'ssl_checkpoints'+'/'+args.exp_name+'/'+'self_main.py.backup')
    os.system('cp model_finetune.py ' + args.pre_path + 'ssl_checkpoints' + '/' + args.exp_name + '/' + 'model_finetune.py.backup')
    os.system('cp util.py '+ args.pre_path +'ssl_checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py '+ args.pre_path +'ssl_checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')
    os.system('cp attack.py '+ args.pre_path +'ssl_checkpoints' + '/' + args.exp_name + '/' + 'attack.py.backup')
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
      m.eval()


def train(args, io):

    train_loader = DataLoader(PCData_SSL(name=args.dataset, partition='train', num_points=args.num_points, rotation=args.rotation, angles=args.angles, jigsaw=args.jigsaw, k=args.k1), num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(PCData_SSL(name=args.dataset,partition='test', num_points=args.num_points, rotation=args.rotation, angles=args.angles, jigsaw=args.jigsaw, k=args.k1), num_workers=8,
                             batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'pointnet_rotation':
        model = PointNet_Rotation(args).to(device)
    elif args.model == 'dgcnn_rotation':
        model = DGCNN_Rotation(args).to(device)
    elif args.model == 'pointnet_jigsaw':
        model = PointNet_Jigsaw(args).to(device)
    elif args.model == 'dgcnn_jigsaw':
        model = DGCNN_Jigsaw(args).to(device)
    elif args.model == 'deepsym_jigsaw':
        model = DeepSym_Jigsaw(args).to(device)
    elif args.model == 'deepsym_rotation':
        model = DeepSym_Rotation(args).to(device)
    elif args.model == 'pct_jigsaw':
        model = Pct_Jigsaw(args).to(device)
    elif args.model == 'pct_rotation':
        model = Pct_Rotation(args).to(device)
    else:
        raise Exception("Not implemented")


    print(str(model))

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
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
    elif args.scheduler == 'pct':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)

    criterion = cal_loss


    best_test_acc = 0
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        count = 0.0
        model.train()

        if args.rotation:
            train_loss_rotation = 0.0
            train_pred_rotation = []
            train_true_rotation = []
        if args.jigsaw:
            train_loss_jigsaw = 0.0
            train_pred_jigsaw = []
            train_true_jigsaw = []

        for aug_data, aug_label in train_loader:
            # print(rotated_data.shape)
            # print(rotation_label.shape)
            # data, label = data.to(device), label.to(device).squeeze()
            batch_size, N, C = aug_data.size()

            aug_data = aug_data.permute(0, 2, 1)

            if args.rotation:
                rotated_data, rotation_label = aug_data.to(device).float(), aug_label.to(device).squeeze()
                if args.adversarial:
                    rotated_data = attack.pgd_attack(model,rotated_data,rotation_label,eps=args.eps,alpha=args.alpha,iters=args.train_iter,mixup=False) 
                    model.train()
                opt.zero_grad()
                logits_rotation,_,trans_feat = model(rotated_data)
                loss_rotation = criterion(logits_rotation,trans_feat,rotation_label)
                loss_rotation.backward()
                opt.step()
                # preds = logits.max(dim=1)[1]
                count += batch_size 

                preds_rotation = logits_rotation.max(dim=1)[1]
                train_loss_rotation += loss_rotation.item() * batch_size
                train_true_rotation.append(rotation_label.cpu().numpy())
                train_pred_rotation.append(preds_rotation.detach().cpu().numpy())    

            elif args.jigsaw:             
                jigsaw_data, jigsaw_label = aug_data.to(device).float(), aug_label.to(device).squeeze().long()
                if args.adversarial:
                    pass
                opt.zero_grad()
                logits_jigsaw,_,_ = model(jigsaw_data)
                logits_jigsaw = logits_jigsaw.view(-1,args.k1**3)
                jigsaw_label = jigsaw_label.view(-1,1)[:,0]
                loss_jigsaw = F.nll_loss(logits_jigsaw,jigsaw_label)
                loss_jigsaw.backward()
                opt.step()
                count += batch_size   

                preds_jigsaw = logits_jigsaw.max(dim=1)[1]
                train_loss_jigsaw += loss_jigsaw.item() * batch_size
                train_true_jigsaw.append(jigsaw_label.cpu().numpy())
                train_pred_jigsaw.append(preds_jigsaw.detach().cpu().numpy())



        if args.rotation:
            train_true_rotation = np.concatenate(train_true_rotation)
            train_pred_rotation = np.concatenate(train_pred_rotation)
        if args.jigsaw:
            train_true_jigsaw = np.concatenate(train_true_jigsaw)
            train_pred_jigsaw = np.concatenate(train_pred_jigsaw)
        
        if args.rotation:
            outstr = 'Train %d, loss_rotation: %.6f, train_rotation acc: %.6f, train_rotation avg acc: %.6f' % (epoch,
                                                                                     train_loss_rotation*1.0/count,
                                                                                     metrics.accuracy_score(
                                                                                         train_true_rotation, train_pred_rotation),
                                                                                     metrics.balanced_accuracy_score(
                                                                                         train_true_rotation, train_pred_rotation)

                                                                                     )
        elif args.jigsaw:
            outstr = 'Train %d, loss_jigsaw: %.6f, train_jigsaw acc: %.6f, train_jigsaw avg acc: %.6f' % (epoch,
                                                                                     train_loss_jigsaw*1.0/count,
                                                                                     metrics.accuracy_score(
                                                                                         train_true_jigsaw, train_pred_jigsaw),
                                                                                     metrics.balanced_accuracy_score(
                                                                                         train_true_jigsaw, train_pred_jigsaw)

                                                                                     )

        io.cprint(outstr)
        scheduler.step()
        
        test(args,io,model=model, dataloader = test_loader)
        # io.cprint(outstr)

        # test_train(args,io,model=model, dataloader = train_loader)
        # # io.cprint(outstr)

        if epoch % 10 == 0 or epoch == 249:
            # adversarial(args,io,model=model, dataloader = test_loader)
            # io.cprint(outstr)

            torch.save(model.state_dict(), args.pre_path + 'ssl_checkpoints/%s/models/model_epoch%d.t7' % (args.exp_name,epoch))
    return model

def test(args, io,model=None, dataloader=None):

    if dataloader == None:
        test_loader = DataLoader(PCData_SSL(name=args.dataset,partition='test', num_points=args.num_points, rotation=args.rotation, angles=args.angles, jigsaw=args.jigsaw, k=args.k1), num_workers=8,
                             batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    else:
        test_loader = dataloader

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if model is None:
        if args.model == 'pointnet_rotation':
            model = PointNet_Rotation(args).to(device)
        elif args.model == 'dgcnn_rotation':
            model = DGCNN_Rotation(args).to(device)
        elif args.model == 'pointnet_jigsaw':
            model = PointNet_Jigsaw(args).to(device)
        elif args.model == 'dgcnn_jigsaw':
            model = DGCNN_Jigsaw(args).to(device)
        elif args.model == 'deepsym_jigsaw':
            model = DeepSym_Jigsaw(args).to(device)
        elif args.model == 'deepsym_rotation':
            model = DeepSym_Rotation(args).to(device)
        elif args.model == 'pct_jigsaw':
            model = Pct_Jigsaw(args).to(device)
        elif args.model == 'pct_rotation':
            model = Pct_Rotation(args).to(device)
        else:
            raise Exception("Not implemented")
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    for data, label in test_loader:

        data, label = data.to(device).float(), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        logits,trans,trans_feat = model(data)
        if args.jigsaw:
            logits = logits.view(-1,args.k1**3)
            label = label.view(-1,1)[:,0]

        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    io.cprint(outstr)


def adversarial(args,io,model=None, dataloader=None):

    if dataloader == None:
        test_loader = DataLoader(PCData_SSL(name=args.dataset,partition='test', num_points=args.num_points, rotation=args.rotation, angles=args.angles, jigsaw=args.jigsaw, k=args.k1), num_workers=8,
                             batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    else:
        test_loader = dataloader

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if model is None:
        if args.model == 'pointnet_rotation':
            model = PointNet_Rotation(args).to(device)
        elif args.model == 'dgcnn_rotation':
            model = DGCNN_Rotation(args).to(device)
        elif args.model == 'pointnet_jigsaw':
            model = PointNet_Jigsaw(args).to(device)
        elif args.model == 'dgcnn_jigsaw':
            model = DGCNN_Jigsaw(args).to(device)
        elif args.model == 'deepsym_jigsaw':
            model = DeepSym_Jigsaw(args).to(device)
        elif args.model == 'deepsym_rotation':
            model = DeepSym_Rotation(args).to(device)
        elif args.model == 'pct_jigsaw':
            model = Pct_Jigsaw(args).to(device)
        elif args.model == 'pct_rotation':
            model = Pct_Rotation(args).to(device)
        else:
            raise Exception("Not implemented")
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(args.model_path))

    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    for data, label in test_loader:
        data, label = data.to(device).float(), label.to(device).squeeze()
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


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--pre_path', type=str, default='./', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet_rotation', 'dgcnn_rotation', 'pointnet_jigsaw', 'dgcnn_jigsaw', 'deepsym_jigsaw', 'deepsym_rotation','pct_rotation','pct_jigsaw'],
                        help='Model to use, [pointnet, dgcnn]')
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
                        help='random seed (default: 1)')
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
    parser.add_argument('--jigsaw',type=bool,default=False,
                        help="Whether to use jigsaw")
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--angles',type=int,default=6,
                        help="How many angles in rotation based ssl")
    parser.add_argument('--k1', type=int, default=2, metavar='N',
                        help='Hyper-parameter k1')
    parser.add_argument('--scheduler',type=str,default='default',
                        help="Which lr scheduler to use")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    _init_()
    print(args.adversarial)
    io = IOStream(args.pre_path + 'ssl_checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
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
        # TEST_ITER=args.test_iter
        model=train(args,io)
        end = time.time()
        io.cprint("Training took %.6f hours" % ((end - start)/3600))
    else:
        pass
        # EPS=args.eps
        # ALPHA=args.alpha
        # TEST_ITER=args.test_iter
        # adversarial(args,io,model=model)
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
