'''
Description: 
Autor: Jiachen Sun
Date: 2021-04-15 14:14:20
LastEditors: Jiachen Sun
LastEditTime: 2021-04-18 18:05:12
'''
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
        if args.model == 'dgcnn':
            model1 = DGCNN(args,output_channels=output_channel).to(device)
            model2 = DGCNN(args,output_channels=output_channel).to(device)
            model3 = DGCNN(args,output_channels=output_channel).to(device)
        elif args.model == 'pointnet_simple':
            model1 = PointNet_Simple(args,output_channels=output_channel).to(device)
            model2 = PointNet_Simple(args,output_channels=output_channel).to(device)
            model3 = PointNet_Simple(args,output_channels=output_channel).to(device)
        elif args.model == 'pct':
            model1 = Pct(args,output_channels=output_channel).to(device)
            model2 = Pct(args,output_channels=output_channel).to(device)
            model3 = Pct(args,output_channels=output_channel).to(device)
        else:
            raise Exception("Not implemented")

        model1 = nn.DataParallel(model1)
        model1.load_state_dict(torch.load(args.model_path1 + '/model_epoch' + str(args.epochs) + '.t7'))
        model2 = nn.DataParallel(model2)
        model2.load_state_dict(torch.load(args.model_path2 + '/model_epoch' + str(args.epochs) + '.t7'))
        model3 = nn.DataParallel(model3)
        model3.load_state_dict(torch.load(args.model_path3 + '/model_epoch' + str(args.epochs) + '.t7'))

    model1 = model1.eval()
    model2 = model2.eval()
    model3 = model3.eval()

    # if args.attack == 'apgd':
    #     apgd = attack.APGDAttack(model1,n_iter=args.test_iter,eps=args.eps,seed=args.seed)
    # elif args.attack == 'apgd_margin':
    #     apgd = attack.APGDAttack(model1,n_iter=args.test_iter,loss='ce_margin',eps=args.eps,seed=args.seed)

    test_true = []
    test_pred = []

    total = args.total
    counter = 0
    
    for data, label,_,_ in test_loader:
        data, label = data.to(device).float(), label.to(device).long().squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]

        if args.attack == 'pgd':
            adv_data = attack.pgd_attack_ensemble(model1,model2,model3,data,label,eps=args.eps,alpha=args.alpha,iters=args.test_iter,repeat=1,mixup=False)
        # elif args.attack == 'pgd_margin':
        #     adv_data = attack.pgd_attack_margin(model1,data,label,eps=args.eps,alpha=args.alpha,iters=args.test_iter,repeat=1,mixup=False)
        # elif args.attack == 'nattack':
        #     adv_data = attack.nattack(model1,data,label,eps=args.eps,alpha=args.alpha,iters=args.test_iter,variance=0.1,samples=args.samples)
        # elif args.attack == 'spsa':
        #     adv_data = attack.spsa(model1,data,label,eps=args.eps,alpha=args.alpha,iters=args.test_iter,samples=args.samples)
        # elif args.attack == 'nes':
        #     adv_data = attack.nes(model1,data,label,eps=args.eps,alpha=args.alpha,iters=args.test_iter,variance=0.001,samples=args.samples)
        # elif args.attack == 'evolution':
        #     adv_data = attack.evolution(model1,data,label,eps=args.eps,iters=args.test_iter,variance=0.005,samples=args.samples,k=args.samples // 4)
        # elif args.attack == 'apgd' or args.attack == 'apgd_margin':
        #     _,adv_data = apgd.perturb(data,label)
        # elif args.attack == 'mim':
        #     adv_data = attack.mim(model1,data,label,eps=args.eps,alpha=args.alpha,iters=args.test_iter,repeat=1,mixup=False)
        # elif args.attack == 'mim_margin':
        #     adv_data = attack.mim_margin(model1,data,label,eps=args.eps,alpha=args.alpha,iters=args.test_iter,repeat=1,mixup=False)
            
        logits1,_,_ = model1(adv_data)
        logits2,_,_ = model2(adv_data)
        logits3,_,_ = model3(adv_data)

        # logits1 = logits1.max(dim = 1)[1]
        # logits2 = logits2.max(dim = 1)[1]
        # logits3 = logits3.max(dim = 1)[1]

        tmax1 = logits1.max(-1, keepdim=True)[0]
        tmax2 = logits2.max(-1, keepdim=True)[0]
        tmax3 = logits3.max(-1, keepdim=True)[0]

        output1 = torch.where(logits1 == tmax1, 1, 0)
        output2 = torch.where(logits2 == tmax2, 1, 0)
        output3 = torch.where(logits3 == tmax3, 1, 0)

        logits = output1 + output2 + output3

        # logits = torch.stack([logits1,logits2,logits3])
        # logits = torch.max(logits,dim=0)[0]

        val,preds = logits.max(dim=1)
        # if val == 1:
        #     logits = logits1 + logits2 + logits3
        #     val,preds = logits.max(dim=1)            

        counter += batch_size
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
        if counter >= total:
            break
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)

    # avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred1)
    outstr = ' Adversarial :: ADV_test acc: %.6f '  %(test_acc)
    io.cprint(args.attack + outstr)
    return 

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn', 'pointnet_simple', 'pct', 'deepsym'],
                        help='Model to use, [pointnet, dgcnn pointnet_simple]')
    parser.add_argument('--pre_path', type=str, default='./', metavar='N',
                        help='Name of the pre path')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='which epoch to evaluate')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 0)')
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
    parser.add_argument('--alpha',type=float,default=0.005,
                        help="Adversarial training perturbation step size")
    parser.add_argument('--test_iter',type=int,default=200,
                        help="Number of steps taken to create adversarial test inputs")
    parser.add_argument('--total',type=int,default=1000,
                        help="Number of samples to evaluate")
    parser.add_argument('--adversarial',type=bool,default=False,
                        help="Whether to use adversarial examples")
    parser.add_argument('--gpu',type=str,default='0',
                        help="Which gpu to use")
    parser.add_argument('--model_path1', type=str, default='', metavar='N',
                        help='Pretrained model path 1')
    parser.add_argument('--model_path2', type=str, default='', metavar='N',
                        help='Pretrained model path 2')
    parser.add_argument('--model_path3', type=str, default='', metavar='N',
                        help='Pretrained model path 3')
    parser.add_argument('--attack', type=str, default='pgd', metavar='N',
                        help='Attack method')
    parser.add_argument('--samples', type=int, default=64, 
                        help='black box samples')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    _init_()

    print(args.adversarial)
    io = IOStream(args.pre_path+'finetune_checkpoints/' + args.exp_name + '/run_' + str(args.epochs) + '_' + args.attack + '_' + str(args.test_iter) + '.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    adversarial(args=args,io=io)