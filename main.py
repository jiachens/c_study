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
from data import ModelNet40, ATTA_ModelNet40, ModelNet40_Jigsaw
from model import PointNet, PointNet_2, DGCNN, SetTransformer, knn, PointNet_3, PointNet_Jigsaw
import numpy as np
from torch.utils.data import DataLoader
import sys
sys.path.append("./emd/")
import emd_module
from util import cal_loss, IOStream, cross_entropy_with_probs,trades_loss
import sklearn.metrics as metrics
import attack
import time
EPS=0.05
ALPHA=0.01
TRAIN_ITER=7
TEST_ITER=7

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp main.py checkpoints'+'/'+args.exp_name+'/'+'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')
    os.system('cp attack.py checkpoints' + '/' + args.exp_name + '/' + 'attack.py.backup')


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
      m.eval()


def train(args, io):
    if not args.jigsaw:
        train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points, rotation=args.rotation, angles=args.angles), num_workers=8,
                                  batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=8,
                                 batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    else:
        train_loader = DataLoader(ModelNet40_Jigsaw(partition='train', num_points=args.num_points, jigsaw=args.jigsaw, k=args.k1), num_workers=8,
                                  batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(ModelNet40_Jigsaw(partition='test', num_points=args.num_points), num_workers=8,
                                 batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'pointnet':
        model = PointNet(args).to(device)
    elif args.model == 'pointnet_2':
        model = PointNet_2(args).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN(args).to(device)
    elif args.model == 'set_transformer':
        model = SetTransformer(args).to(device)
    elif args.model == 'pointnet_3':
        model = PointNet_3(args).to(device)
    elif args.model == 'pointnet_jigsaw':
        model = PointNet_Jigsaw(args).to(device)
    else:
        raise Exception("Not implemented")

    # for name,m in model.named_modules():
    #     if name == 'stn.fc3' or name == 'fstn.fc3':
    #         nn.init.constant_(m.weight, 0.)
    #     # elif isinstance(m, (nn.Conv1d, nn.Linear)):
    #     #     nn.init.xavier_uniform_(m.weight)
    #     #     nn.init.constant_(m.bias, 0.)
    #     elif isinstance(m, nn.BatchNorm1d):
    #         nn.init.constant_(m.weight, 1.)
    #         nn.init.constant_(m.bias, 0.)

    print(str(model))

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr)# , weight_decay=1e-4)

    
    #scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    scheduler = StepLR(opt, 20, 0.7)
    #scheduler = MultiStepLR(opt, milestones=[100,150,200], gamma=0.1)

    if args.mixup:
        criterion = cross_entropy_with_probs
    else:
        criterion = cal_loss

    if args.mixup:
        EMD = emd_module.emdModule()

    best_test_acc = 0
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################

        # if args.model == 'pointnet':
        #     if epoch % 20 == 0:
        #         for layer in model.modules():
        #             if type(layer) == torch.nn.modules.batchnorm.BatchNorm1d:
        #                 layer.momentum = max(0.01, 0.5**((epoch // 20) + 1))
        #             # if type(layer) == model.STN3d or type(layer) == model.STNkd:
        #             #     for sublayer in layer:
        #             #         if type(sublayer) == torch.nn.modules.batchnorm.BatchNorm1d:
        #             #             sublayer.momentum = np.max(0.01, 0.5**((epoch // 20) + 1))
        # print(opt.param_groups[0]['lr'])

        train_loss = 0.0
        count = 0.0
        model.train()
        #model.apply(set_bn_eval)
        train_pred = []
        train_true = []
        if args.rotation:
            train_loss_rotation = 0.0
            train_pred_rotation = []
            train_true_rotation = []
        if args.jigsaw:
            train_loss_jigsaw = 0.0
            train_pred_jigsaw = []
            train_true_jigsaw = []

        for data, label, aug_data, aug_label in train_loader:
            # print(rotated_data.shape)
            # print(rotation_label.shape)
            data, label = data.to(device), label.to(device).squeeze()
            batch_size, N, C = data.size()
            # print(batch_size,N,C)
            if args.mixup:
                idx_minor = torch.randperm(batch_size)
                mixrates = (0.5 - np.abs(np.random.beta(0.4, 0.4, batch_size) - 0.5))
                label_main = label
                label_minor = label[idx_minor]
                label_new = torch.zeros(batch_size, 40)
                for i in range(batch_size):
                    if label_main[i] == label_minor[i]: # same label
                        label_new[i][label_main[i]] = 1.0
                    else:
                        label_new[i][label_main[i]] = 1 - mixrates[i]
                        label_new[i][label_minor[i]] = mixrates[i]
                label = label_new

                data_minor = data[idx_minor]
                mix_rate = torch.tensor(mixrates).cuda().float()
                mix_rate = mix_rate.unsqueeze_(1).unsqueeze_(2)

                mix_rate_expand_xyz = mix_rate.expand(data.shape)
                _, ass = EMD(data, data_minor, 0.005, 300)
                ass = ass.long()
                for i in range(batch_size):
                    data_minor[i] = data_minor[i][ass[i]]
                data = data * (1 - mix_rate_expand_xyz) + data_minor * mix_rate_expand_xyz
                
            if args.cutout:
                idx = np.random.randint(args.num_points,size=batch_size)
                for i in range(batch_size):
                    picked = data[i][idx[i]]
                    dist = torch.sum((data[i] - picked)**2, dim=1, keepdim=True)
                    idx = dist.topk(k=100, largest=False, dim=0)[1]
                    data[i,idx.squeeze()] = 0

            # np.save('./test.npy',data.cpu().numpy())

            data = data.permute(0, 2, 1)
            if args.rotation or args.jigsaw:
                aug_data = aug_data.permute(0, 2, 1)

            if args.trades:
                opt.zero_grad()
                # calculate robust loss - TRADES loss
                loss = trades_loss(model=model,
                                   x_natural=data,
                                   y=label,
                                   optimizer=opt,
                                   step_size=args.alpha,
                                   epsilon=args.eps,
                                   perturb_steps=args.test_iter,
                                   beta=1.0,
                       distance='l_inf')
                loss.backward()
                opt.step()
                count += batch_size
                train_loss += loss.item() * batch_size
            else:
                if not args.rotation and not args.jigsaw:
                    if args.adversarial:
                        data = attack.pgd_attack(model,data,label,eps=EPS,alpha=ALPHA,iters=TRAIN_ITER,mixup=args.mixup) 
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
                elif args.rotation:
                    rotated_data, rotation_label = aug_data.to(device).float(), aug_label.to(device).squeeze()
                    if args.adversarial:
                        data = attack.pgd_attack(model,data,label,eps=EPS,alpha=ALPHA,iters=TRAIN_ITER,mixup=args.mixup) 
                        model.train()
                    opt.zero_grad()
                    logits,trans,trans_feat = model(data,rotation = False)
                    loss = criterion(logits, trans_feat, label)
                    logits_rotation,_,_ = model(rotated_data,rotation = True)
                    loss_rotation = criterion(logits_rotation,None,rotation_label)
                    loss_total = loss + args.lambda1 * loss_rotation
                    loss_total.backward()
                    opt.step()
                    preds = logits.max(dim=1)[1]
                    count += batch_size
                    train_loss += loss.item() * batch_size
                    train_true.append(label.cpu().numpy())
                    train_pred.append(preds.detach().cpu().numpy())    

                    preds_jigsaw = logits_jigsaw.max(dim=1)[1]
                    train_loss_rotation += loss_rotation.item() * batch_size
                    train_true_rotation.append(rotation_label.cpu().numpy())
                    train_pred_rotation.append(preds_rotation.detach().cpu().numpy())    

                elif args.jigsaw:             
                    jigsaw_data, jigsaw_label = aug_data.to(device).float(), aug_label.to(device).squeeze().long()
                    if args.adversarial:
                        data = attack.pgd_attack(model,data,label,eps=EPS,alpha=ALPHA,iters=TRAIN_ITER,mixup=args.mixup) 
                        model.train()
                    opt.zero_grad()
                    logits,trans,trans_feat = model(data,jigsaw = False)
                    loss = criterion(logits, trans_feat, label)
                    logits_jigsaw,_,_ = model(jigsaw_data,jigsaw = True)
                    logits_jigsaw = logits_jigsaw.view(-1,args.k1**3)
                    jigsaw_label = jigsaw_label.view(-1,1)[:,0]
                    loss_jigsaw = F.nll_loss(logits_jigsaw,jigsaw_label)
                    loss_total = loss + args.lambda1 * loss_jigsaw
                    loss_total.backward()
                    opt.step()
                    preds = logits.max(dim=1)[1]
                    count += batch_size
                    train_loss += loss.item() * batch_size
                    train_true.append(label.cpu().numpy())
                    train_pred.append(preds.detach().cpu().numpy())    

                    preds_jigsaw = logits_jigsaw.max(dim=1)[1]
                    train_loss_jigsaw += loss_jigsaw.item() * batch_size
                    train_true_jigsaw.append(jigsaw_label.cpu().numpy())
                    train_pred_jigsaw.append(preds_jigsaw.detach().cpu().numpy())


        if not args.trades:
            train_true = np.concatenate(train_true)
            train_pred = np.concatenate(train_pred)
            if args.rotation:
                train_true_rotation = np.concatenate(train_true_rotation)
                train_pred_rotation = np.concatenate(train_pred_rotation)
            if args.jigsaw:
                train_true_jigsaw = np.concatenate(train_true_jigsaw)
                train_pred_jigsaw = np.concatenate(train_pred_jigsaw)
        if args.mixup or args.trades:
            outstr = 'Train %d, loss: %.6f' % (epoch, train_loss*1.0/count)
        elif args.rotation:
            outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f loss_rotation: %.6f, train_rotation acc: %.6f, train_rotation avg acc: %.6f' % (epoch,
                                                                                     train_loss*1.0/count,
                                                                                     metrics.accuracy_score(
                                                                                         train_true, train_pred),
                                                                                     metrics.balanced_accuracy_score(
                                                                                         train_true, train_pred),
                                                                                     train_loss_rotation*1.0/count,
                                                                                     metrics.accuracy_score(
                                                                                         train_true_rotation, train_pred_rotation),
                                                                                     metrics.balanced_accuracy_score(
                                                                                         train_true_rotation, train_pred_rotation)

                                                                                     )
        elif args.jigsaw:
            outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f loss_jigsaw: %.6f, train_jigsaw acc: %.6f, train_jigsaw avg acc: %.6f' % (epoch,
                                                                                     train_loss*1.0/count,
                                                                                     metrics.accuracy_score(
                                                                                         train_true, train_pred),
                                                                                     metrics.balanced_accuracy_score(
                                                                                         train_true, train_pred),
                                                                                     train_loss_jigsaw*1.0/count,
                                                                                     metrics.accuracy_score(
                                                                                         train_true_jigsaw, train_pred_jigsaw),
                                                                                     metrics.balanced_accuracy_score(
                                                                                         train_true_jigsaw, train_pred_jigsaw)

                                                                                     )
        else:
            outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                     train_loss*1.0/count,
                                                                                     metrics.accuracy_score(
                                                                                         train_true, train_pred),
                                                                                     metrics.balanced_accuracy_score(
                                                                                         train_true, train_pred))
        io.cprint(outstr)
        scheduler.step()
        
        test(args,io,model=model, dataloader = test_loader)
        # io.cprint(outstr)

        test_train(args,io,model=model, dataloader = train_loader)
        # io.cprint(outstr)

        if epoch % 10 == 0:
           adversarial(args,io,model=model, dataloader = test_loader)
           # io.cprint(outstr)

        torch.save(model.state_dict(), 'checkpoints/%s/models/model_epoch%d.t7' % (args.exp_name,epoch))
    return model

def test(args, io,model=None, dataloader=None):

    if dataloader == None:
        test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points),
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    else:
        test_loader = dataloader

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if model is None:
        if args.model == 'pointnet':
            model = PointNet(args).to(device)
        elif args.model == 'pointnet_2':
            model = PointNet_2(args).to(device)
        elif args.model == 'dgcnn':
            model = DGCNN(args).to(device)
        elif args.model == 'set_transformer':
            model = SetTransformer(args).to(device)
        elif args.model == 'pointnet_3':
            model = PointNet_3(args).to(device)
        else:
            raise Exception("Not implemented")
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    for data, label, _, _ in test_loader:

        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        logits,trans,trans_feat = model(data)
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    io.cprint(outstr)

def test_train(args, io,model=None, dataloader=None):
    if dataloader == None:
        test_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points),
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    else:
        test_loader = dataloader

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if model is None:
        if args.model == 'pointnet':
            model = PointNet(args).to(device)
        elif args.model == 'pointnet_2':
            model = PointNet_2(args).to(device)
        elif args.model == 'dgcnn':
            model = DGCNN(args).to(device)
        elif args.model == 'set_transformer':
            model = SetTransformer(args).to(device)
        elif args.model == 'pointnet_3':
            model = PointNet_3(args).to(device)
        else:
            raise Exception("Not implemented")
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    for data, label, _, _ in test_loader:

        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        logits,trans,trans_feat = model(data)
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test_train :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    io.cprint(outstr)

def adversarial(args,io,model=None, dataloader=None):
    if dataloader == None:
        test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points),
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    else:
        test_loader = dataloader

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if model is None:
        if args.model == 'pointnet':
            model = PointNet(args).to(device)
        elif args.model == 'pointnet_2':
            model = PointNet_2(args).to(device)
        elif args.model == 'dgcnn':
            model = DGCNN(args).to(device)
        elif args.model == 'set_transformer':
            model = SetTransformer(args).to(device)
        elif args.model == 'pointnet_3':
            model = PointNet_3(args).to(device)
        else:
            raise Exception("Not implemented")
        model = nn.DataParallel(model) 
        model.load_state_dict(torch.load(args.model_path))

    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    for data, label, _, _ in test_loader:
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        adv_data = attack.pgd_attack(model,data,label,eps=EPS,alpha=ALPHA,iters=TEST_ITER,repeat=1,mixup=False)
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
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn','set_transformer', 'pointnet_2', 'pointnet_3', 'pointnet_jigsaw'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
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
    parser.add_argument('--atta_reset',type=int,default=25,
                        help="Number of steps epochs before resetting ATTA examples")
    parser.add_argument('--adversarial',type=bool,default=False,
                        help="Whether to use adversarial examples")
    parser.add_argument('--fspool_local',type=bool,default=False,
                        help="Whether to use FSPool locally, defaults to max pool")
    parser.add_argument('--fspool_global',type=bool,default=False,
                        help="Whether to use FSPool globally, defaults to max pool")
    parser.add_argument('--mlppool_global',type=bool,default=False,
                        help="Whether to use MLPPool globally, defaults to max pool")
    parser.add_argument('--atta',type=bool,default=False,
                        help="Whether to train using ATTA")
    parser.add_argument('--set_transformer_maxpool',type=bool,default=False,
                        help="Whether to use maxpool for set_transformer")
    parser.add_argument('--mixup',type=bool,default=False,
                        help="Whether to use mixup")
    parser.add_argument('--cutout',type=bool,default=False,
                        help="Whether to use cutout")
    parser.add_argument('--trades',type=bool,default=False,
                        help="Whether to use trades")
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
    parser.add_argument('--lambda1',type=float,default=1.,
                        help="Hyper-parameter lambda")
    parser.add_argument('--k1', type=int, default=2, metavar='N',
                        help='Hyper-parameter k1')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    _init_()
    print(args.adversarial)
    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
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
        if args.adversarial:
            EPS=args.eps
            ALPHA=args.alpha
            TRAIN_ITER=args.train_iter
            TEST_ITER=args.test_iter
            if args.atta:
                model=atta_train(args, io)
            else:
                model=train(args,io)
        else:
            model = train(args,io)
        end = time.time()
        io.cprint("Training took %.6f hours" % ((end - start)/3600))
    else:
        EPS=args.eps
        ALPHA=args.alpha
        TEST_ITER=args.test_iter
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
