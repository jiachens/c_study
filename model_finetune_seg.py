'''
Description: 
Autor: Jiachen Sun
Date: 2021-04-04 15:30:04
LastEditors: Jiachen Sun
LastEditTime: 2021-04-05 11:27:55
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

class DGCNN_Seg(nn.Module):
    def __init__(self, args, output_channels):
        super(DGCNN_Seg, self).__init__()
        self.args = args
        self.seg_num_all = output_channels
        self.k = args.k
        # self.transform_net = Transform_Net(args)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.bn206 = nn.BatchNorm1d(64)
        self.bn207 = nn.BatchNorm1d(256)
        self.bn208 = nn.BatchNorm1d(256)
        self.bn209 = nn.BatchNorm1d(128)

        self.conv206 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   self.bn206,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv207 = nn.Sequential(nn.Conv1d(1088 + 512, 256, kernel_size=1, bias=False),
                                   self.bn207,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp201 = nn.Dropout(p=args.dropout)
        self.conv208 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn208,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp202 = nn.Dropout(p=args.dropout)
        self.conv209 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn209,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2010 = nn.Conv1d(128, self.seg_num_all, kernel_size=1, bias=False)
        

    def forward(self, x, l):
        batch_size = x.size(0)
        num_points = x.size(2)
        
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1, 1)
        # x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1, 1)
        x = x1


        l = l.view(batch_size, -1, 1)           
        l = self.conv206(l)                      

        x = torch.cat((x, l), dim=1)           
        x = x.repeat(1, 1, num_points)          

        x = torch.cat((x, x1, x2, x3, x4), dim=1)   
        x = self.conv207(x)    
        x = self.dp201(x)                  
        x = self.conv208(x)                      
        x = self.dp202(x)
        x = self.conv209(x)                     
        x = self.conv2010(x)                      
        
        return x

class PointNet_Simple_Seg(nn.Module):
    def __init__(self, args, output_channels):
        super(PointNet_Simple_Seg, self).__init__()
        self.args = args
        self.output_channels = output_channels
        #self.stn = STN3d()
        #self.fstn = STNkd(k=64)
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64,eps=1e-03)
        self.bn2 = nn.BatchNorm1d(64,eps=1e-03)
        self.bn3 = nn.BatchNorm1d(64,eps=1e-03)
        self.bn4 = nn.BatchNorm1d(128,eps=1e-03)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv206 = nn.Conv1d(64 + 64 + args.emb_dims, 512, 1, bias=False)
        self.conv207 = nn.Conv1d(512, 256, 1, bias=False)
        self.conv208 = nn.Conv1d(256, 128, 1, bias=False)
        self.conv209 = nn.Conv1d(128, self.output_channels, 1, bias=False)
        self.bn201 = nn.BatchNorm1d(512,eps=1e-03)
        self.bn202 = nn.BatchNorm1d(256,eps=1e-03)
        self.bn203 = nn.BatchNorm1d(128,eps=1e-03)

        self.conv2010 = nn.Conv1d(16, 64, 1, bias=False)
        self.bn204 = nn.BatchNorm1d(64,eps=1e-03)

    def forward(self, x, l):
        batchsize = x.size()[0]

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        pointfeat = x
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).view(batchsize, -1, 1)

        l = l.view(batchsize, -1, 1)           
        l = F.relu(self.bn204(self.conv2010(l)))                      
        x = torch.cat((x, l), dim=1)
        
        x = x.repeat(1, 1, self.args.num_points)
        x = torch.cat([x, pointfeat], 1)
        x = F.relu(self.bn201(self.conv206(x)))
        x = F.relu(self.bn202(self.conv207(x)))
        x = F.relu(self.bn203(self.conv208(x)))
        x = self.conv209(x)
        # x = x.transpose(2,1).contiguous()
        # x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, self.output_channels, self.args.num_points)
        
        return x

class Pct_Seg(nn.Module):
    def __init__(self, args, output_channels):
        super(Pct_Seg, self).__init__()
        self.args = args
        self.output_channels = output_channels
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

        self.seq1 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.seq2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.pt_last = Point_Transformer_Last(args)

        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))

        self.conv_fuse201 = nn.Sequential(nn.Conv1d(1024, 256, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))
        self.dp201 = nn.Dropout(p=args.dropout)
        
        self.conv_fuse202 = nn.Sequential(nn.Conv1d(1024 + 256, 256, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))

        self.conv_fuse203 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))
        
        self.conv_fuse204 = nn.Conv1d(256, self.output_channels, kernel_size=1, bias=False)

        # self.linear1 = nn.Linear(1024, 512, bias=False)
        # self.bn6 = nn.BatchNorm1d(512)
        # self.dp1 = nn.Dropout(p=args.dropout)
        # self.linear2 = nn.Linear(512, 256)
        # self.bn7 = nn.BatchNorm1d(256)
        # self.dp2 = nn.Dropout(p=args.dropout)
        # self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        # B, D, N
        x = F.relu(self.bn2(self.conv2(x)))
        # x = x.permute(0, 2, 1)

        x = get_graph_feature(x, k=32)
        x = self.seq1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=32)
        x = self.seq2(x)
        feature_1 = x.max(dim=-1, keepdim=False)[0]

        x = self.pt_last(feature_1)
        x = torch.cat([x, feature_1], dim=1)
        x = self.conv_fuse(x)
        x2 = self.conv_fuse201(x)

        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1, 1).repeat(1, 1, self.args.num_points)

        x = torch.cat([x, x2], 1)
        x = self.conv_fuse202(x)
        x = self.dp201(x)
        x = self.conv_fuse203(x)
        x = self.conv_fuse204(x)
        # x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        # x = self.dp1(x)
        # x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        # x = self.dp2(x)
        # x = self.linear3(x)
        x = x.view(batch_size, self.output_channels, self.args.num_points)
        return x


class Point_Transformer_Last(nn.Module):
    def __init__(self, args, channels=256):
        super(Point_Transformer_Last, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

    def forward(self, x):
        # 
        # b, 3, npoint, nsample  
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample 
        # permute reshape
        batch_size, _, N = x.size()

        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x

class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # b, n, c
        x_q = self.q_conv(x).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        # b, n, n
        energy = torch.bmm(x_q, x_k)

        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # b, c, n
        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x