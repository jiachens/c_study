'''
Description: 
Autor: Jiachen Sun
Date: 2021-03-23 21:06:56
LastEditors: Jiachen Sun
LastEditTime: 2021-03-24 22:53:19
'''

import os
import sys
import copy
import math
import numpy as np
np.random.seed(666)
import torch
torch.manual_seed(666)
torch.cuda.manual_seed_all(666)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import torch.nn.functional as F
from util import sample_and_group 
from torch.autograd import Variable
from modules import ISAB, PMA, SAB

class Dual_BN(nn.Module):
    def __init__(self, num_channels,eps=1e-3):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(num_channels,eps=eps)
        self.bn2 = nn.BatchNorm1d(num_channels,eps=eps)

    def forward(self,x,flag = False):
        if flag:
            x = self.bn1(x)
        else:
            x = self.bn2(x)
        return x

class Dual_BN2d(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_channels)
        self.bn_ssl = nn.BatchNorm2d(num_channels)

    def forward(self,x,flag = False):
        if flag:
            x = self.bn_ssl(x)
        else:
            x = self.bn(x)
        return x
def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature

class PointNet_Simple_Combine(nn.Module):
    def __init__(self, args):
        super(PointNet_Simple_Combine, self).__init__()
        self.args = args
        self.k = (args.k1)**3
        self.angles = args.angles
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.dual_bn1 = Dual_BN(64,eps=1e-03)
        self.dual_bn2 = Dual_BN(64,eps=1e-03)
        self.dual_bn3 = Dual_BN(64,eps=1e-03)
        self.dual_bn4 = Dual_BN(128,eps=1e-03)
        self.dual_bn5 = Dual_BN(args.emb_dims)

        self.linear101 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn106 = nn.BatchNorm1d(512,eps=1e-03)
        self.dp101 = nn.Dropout(p=0.3)
        self.linear102 = nn.Linear(512, 256, bias=False)
        self.bn107 = nn.BatchNorm1d(256,eps=1e-03)
        self.dp102 = nn.Dropout(p=0.3)
        self.linear103 = nn.Linear(256,self.angles)

        self.conv106 = nn.Conv1d(64 + args.emb_dims, 512, 1, bias=False)
        self.conv107 = nn.Conv1d(512, 256, 1, bias=False)
        self.conv108 = nn.Conv1d(256, 128, 1, bias=False)
        self.conv109 = nn.Conv1d(128, self.k, 1, bias=False)
        self.bn108 = nn.BatchNorm1d(512,eps=1e-03)
        self.bn109 = nn.BatchNorm1d(256,eps=1e-03)
        self.bn1010 = nn.BatchNorm1d(128,eps=1e-03)

    def forward(self, x, rotation=False):
        batchsize = x.size()[0]
        # trans = self.stn(x,jigsaw)
        # x = x.transpose(2, 1)
        # x = torch.bmm(x,trans)
        # x = x.transpose(2, 1)
        x = F.relu(self.dual_bn1(self.conv1(x),rotation))
        x = F.relu(self.dual_bn2(self.conv2(x),rotation))
        # trans_feat = self.fstn(x,jigsaw)
        # x = x.transpose(2,1)
        # x = torch.bmm(x, trans_feat)
        # x = x.transpose(2,1)
        x = F.relu(self.dual_bn3(self.conv3(x),rotation))
        pointfeat = x
        x = F.relu(self.dual_bn4(self.conv4(x),rotation))
        x = F.relu(self.dual_bn5(self.conv5(x),rotation))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        if rotation:
            x = F.relu(self.bn106(self.linear101(x)))
            x = self.dp101(x)
            x = F.relu(self.bn107(self.linear102(x)))
            x = self.dp102(x)
            x = self.linear103(x)
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, self.args.num_points)
            x = torch.cat([x, pointfeat], 1)
            x = F.relu(self.bn108(self.conv106(x)))
            x = F.relu(self.bn109(self.conv107(x)))
            x = F.relu(self.bn1010(self.conv108(x)))
            x = self.conv109(x)
            x = x.transpose(2,1).contiguous()
            x = F.log_softmax(x.view(-1,self.k), dim=-1)
            x = x.view(batchsize, self.args.num_points, self.k)
        
        return x, None, None

class MAXPOOL(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x.max(dim=-1, keepdim=False)[0]

class DGCNN_Combine(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN_Combine, self).__init__()
        self.args = args
        self.k = args.k
        self.k1 = args.k1**3
        self.angles = args.angles
        # self.FSPool_local=args.fspool_local
        # self.FSPool_global = args.fspool_global
        # self.MLPPool_global = args.mlppool_global
        self.dual_bn1 = Dual_BN2d(64)
        self.dual_bn2 = Dual_BN2d(64)
        self.dual_bn3 = Dual_BN2d(128)
        self.dual_bn4 = Dual_BN2d(256)
        self.dual_bn5 = Dual_BN(args.emb_dims)
        

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        
        self.conv1 = nn.Conv2d(6, 64, kernel_size=1, bias=False)
                                   # self.bn1,

        self.conv2 = nn.Conv2d(64*2, 64, kernel_size=1, bias=False)
                                   # self.bn2,
                                   # nn.LeakyReLU(negative_slope=0.2))

        self.conv3 = nn.Conv2d(64*2, 128, kernel_size=1, bias=False)
                                   # self.bn3,
                                   # nn.LeakyReLU(negative_slope=0.2))

        self.conv4 = nn.Conv2d(128*2, 256, kernel_size=1, bias=False)
                                   # self.bn4,
                                   # nn.LeakyReLU(negative_slope=0.2))

        self.conv5 = nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False)
                                   # self.bn5,
                                   # nn.LeakyReLU(negative_slope=0.2))
        
        
        
        self.linear101 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.dual_bn106 = Dual_BN(512)
        self.dp101 = nn.Dropout(p=args.dropout)
        self.linear102 = nn.Linear(512, 256)
        self.dual_bn107 = Dual_BN(256)
        self.dp102 = nn.Dropout(p=args.dropout)
        self.linear103 = nn.Linear(256, self.angles)


        self.conv106 = nn.Conv1d(64 + args.emb_dims * 2, 512, 1, bias=False)
        self.conv107 = nn.Conv1d(512, 256, 1, bias=False)
        self.conv108 = nn.Conv1d(256, 128, 1, bias=False)
        self.conv109 = nn.Conv1d(128, self.k1, 1, bias=False)
        self.dual_bn108 = Dual_BN(512)
        self.dual_bn109 = Dual_BN(256)
        self.dual_bn1010 = Dual_BN(128)

        self.pool1 = MAXPOOL()
        self.pool2 = MAXPOOL()
        self.pool3 = MAXPOOL()
        self.pool4 = MAXPOOL()
        # if(self.FSPool_global):
        #     self.pool5 = FSPOOL(1024,1024)
        #     self.pool6 = FSPOOL(1024,1024)
        # elif(self.MLPPool_global):
        # self.pool5 = MLPPool(1024,2,1024)
    
    def forward(self, x, rotation=False):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.lrelu(self.dual_bn1(self.conv1(x), rotation))
        x1 = self.pool1(x)

        x = get_graph_feature(x1, k=self.k)
        x = self.lrelu(self.dual_bn2(self.conv2(x), rotation))
        x2 = self.pool2(x)

        pointfeat = x2

        x = get_graph_feature(x2, k=self.k)
        x = self.lrelu(self.dual_bn3(self.conv3(x), rotation))
        x3 = self.pool3(x)

        x = get_graph_feature(x3, k=self.k)
        x = self.lrelu(self.dual_bn4(self.conv4(x), rotation))
        x4 = self.pool4(x)

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.lrelu(self.dual_bn5(self.conv5(x), rotation))

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        if not rotation:
            x = x.view(-1, self.args.emb_dims * 2, 1).repeat(1, 1, self.args.num_points)
            x = torch.cat([x, pointfeat], 1)
            
            x = F.leaky_relu(self.dual_bn108(self.conv106(x),rotation), negative_slope=0.2)
            x = F.leaky_relu(self.dual_bn109(self.conv107(x),rotation), negative_slope=0.2)
            x = F.leaky_relu(self.dual_bn1010(self.conv108(x),rotation), negative_slope=0.2)
            x = self.conv109(x)
            x = x.transpose(2,1).contiguous()
            x = F.log_softmax(x.view(-1,self.k1), dim=-1)
            x = x.view(batch_size, self.args.num_points, self.k1)
        else:
            x = F.leaky_relu(self.dual_bn106(self.linear101(x),rotation), negative_slope=0.2)
            x = self.dp101(x)
            x = F.leaky_relu(self.dual_bn107(self.linear102(x),rotation), negative_slope=0.2)
            x = self.dp102(x)
            x = self.linear103(x)
            
        return x, None, None