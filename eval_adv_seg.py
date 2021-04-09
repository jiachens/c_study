'''
Description: 
Autor: Jiachen Sun
Date: 2021-04-09 17:13:10
LastEditors: Jiachen Sun
LastEditTime: 2021-04-09 17:38:03
'''

from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, StepLR, MultiStepLR, ReduceLROnPlateau
from data import PCData_SSL, PCData, PCData_Jigsaw, ShapeNetPart
from model_finetune_seg import PointNet_Simple_Seg, DGCNN_Seg, Pct_Seg
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

def calculate_shape_IoU(pred_np, seg_np, label, class_choice):
    label = label.squeeze()
    shape_ious = []
    for shape_idx in range(seg_np.shape[0]):
        if not class_choice:
            start_index = index_start[label[shape_idx]]
            num = seg_num[label[shape_idx]]
            parts = range(start_index, start_index + num)
        else:
            parts = range(seg_num[label[0]])
        part_ious = []
        for part in parts:
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            if U == 0:
                iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))
    return shape_ious

seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]

def _init_():
    if not os.path.exists(args.pre_path +'finetune_seg_checkpoints'):
        os.makedirs(args.pre_path +'finetune_seg_checkpoints')
    if not os.path.exists(args.pre_path +'finetune_seg_checkpoints/'+args.exp_name):
        os.makedirs(args.pre_path +'finetune_seg_checkpoints/'+args.exp_name)
    if not os.path.exists(args.pre_path +'finetune_seg_checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs(args.pre_path +'finetune_seg_checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp finetune_main_seg.py '+args.pre_path+'finetune_seg_checkpoints'+'/'+args.exp_name+'/'+'finetune_main_seg.py.backup')
    os.system('cp model_finetune.py '+args.pre_path+'finetune_seg_checkpoints' + '/' + args.exp_name + '/' + 'model_finetune.py.backup')
    os.system('cp util.py '+args.pre_path+'finetune_seg_checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py '+args.pre_path+'finetune_seg_checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')
    os.system('cp attack.py '+args.pre_path+'finetune_seg_checkpoints' + '/' + args.exp_name + '/' + 'attack.py.backup')
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False

    
def adversarial(args,io,model=None, dataloader=None):

    if dataloader == None:
        test_loader = DataLoader(ShapeNetPart(partition='test', num_points=args.num_points, class_choice=args.class_choice), 
                            num_workers=8, batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    else:
        test_loader = dataloader

    device = torch.device("cuda" if args.cuda else "cpu")

    seg_num_all = test_loader.dataset.seg_num_all
    seg_start_index = test_loader.dataset.seg_start_index
    
    #Try to load models
    if model is None:
        if args.model == 'pointnet_simple':
            model = PointNet_Simple_Seg(args,output_channels=seg_num_all).to(device)
        elif args.model == 'dgcnn':
            model = DGCNN_Seg(args,output_channels=seg_num_all).to(device)
        elif args.model == 'pct':
            model = Pct_Seg(args,output_channels=seg_num_all).to(device)
        else:
            raise Exception("Not implemented")
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(args.model_path + '/model_epoch' + str(args.epochs) + '.t7'))


    test_loss = 0.0
    count = 0.0
    model.eval()
    test_true_cls = []
    test_pred_cls = []
    test_true_seg = []
    test_pred_seg = []
    test_label_seg = []
    criterion = cal_loss

    for data, label, seg in test_loader:
        seg = seg - seg_start_index
        label_one_hot = np.zeros((label.shape[0], 16))
        for idx in range(label.shape[0]):
            label_one_hot[idx, label[idx]] = 1
        label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))
        data, label_one_hot, seg = data.to(device), label_one_hot.to(device), seg.to(device)
        data = data.permute(0, 2, 1)

        if args.attack == 'pgd':
            data = attack.pgd_attack_partseg(model,data,seg,label_one_hot,eps=args.eps,number=seg_num_all,alpha=args.alpha,iters=args.train_iter)
        batch_size = data.size()[0]

        seg_pred = model(data,label_one_hot)
        seg_pred = seg_pred.permute(0, 2, 1).contiguous()
        loss = criterion(seg_pred.view(-1, seg_num_all),None, seg.view(-1,1).squeeze())
        pred = seg_pred.max(dim=2)[1]
        count += batch_size
        test_loss += loss.item() * batch_size
        seg_np = seg.cpu().numpy()
        pred_np = pred.detach().cpu().numpy()
        test_true_cls.append(seg_np.reshape(-1))
        test_pred_cls.append(pred_np.reshape(-1))
        test_true_seg.append(seg_np)
        test_pred_seg.append(pred_np)
        test_label_seg.append(label.reshape(-1))
    test_true_cls = np.concatenate(test_true_cls)
    test_pred_cls = np.concatenate(test_pred_cls)
    test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
    test_true_seg = np.concatenate(test_true_seg, axis=0)
    test_pred_seg = np.concatenate(test_pred_seg, axis=0)
    test_label_seg = np.concatenate(test_label_seg)
    test_ious = calculate_shape_IoU(test_pred_seg, test_true_seg, test_label_seg, args.class_choice)
    outstr = ' Adversarial :: loss: %.6f, test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (
                                                                                            test_loss*1.0/count,
                                                                                            test_acc,
                                                                                            avg_per_class_acc,
                                                                                            np.mean(test_ious))
    io.cprint(args.attack + outstr)
    return np.mean(test_ious)

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
    parser.add_argument('--dataset', type=str, default='shapenetpart', metavar='N')
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
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--attack', type=str, default='pgd', metavar='N',
                        help='Attack method')
    parser.add_argument('--class_choice', type=str, default=None, metavar='N',
                        choices=['airplane', 'bag', 'cap', 'car', 'chair',
                                 'earphone', 'guitar', 'knife', 'lamp', 'laptop', 
                                 'motor', 'mug', 'pistol', 'rocket', 'skateboard', 'table'])
    parser.add_argument('--samples', type=int, default=64, 
                        help='black box samples')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    _init_()
    print(args.adversarial)
    io = IOStream(args.pre_path+'finetune_seg_checkpoints/' + args.exp_name + '/run_' + str(args.epochs) + '_' + args.attack + '_' + str(args.test_iter) + '.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    adversarial(args,io,model=None)