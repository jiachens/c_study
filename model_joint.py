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
from torch.autograd import Variable
from util import sample_and_group 

class Dual_BN(nn.Module):
    def __init__(self, num_channels,eps=1e-3):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_channels,eps=eps)
        self.bn_ssl = nn.BatchNorm1d(num_channels,eps=eps)

    def forward(self,x,flag = False):
        if flag:
            x = self.bn_ssl(x)
        else:
            x = self.bn(x)
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

class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = Dual_BN(64,eps=1e-03)
        self.bn2 = Dual_BN(128,eps=1e-03)
        self.bn3 = Dual_BN(1024,eps=1e-03)
        self.bn4 = Dual_BN(512,eps=1e-03)
        self.bn5 = Dual_BN(256,eps=1e-03)


    def forward(self, x, flag):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x), flag))
        x = F.relu(self.bn2(self.conv2(x), flag))
        x = F.relu(self.bn3(self.conv3(x), flag))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x), flag))
        x = F.relu(self.bn5(self.fc2(x), flag))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)

        self.relu = nn.ReLU()

        self.bn1 = Dual_BN(64,eps=1e-03)
        self.bn2 = Dual_BN(128,eps=1e-03)
        self.bn3 = Dual_BN(1024,eps=1e-03)
        self.bn4 = Dual_BN(512,eps=1e-03)
        self.bn5 = Dual_BN(256,eps=1e-03)

        self.k = k

    def forward(self, x, flag):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x), flag))
        x = F.relu(self.bn2(self.conv2(x), flag))
        x = F.relu(self.bn3(self.conv3(x), flag))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x), flag))
        x = F.relu(self.bn5(self.fc2(x), flag))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
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

class MLPPool(nn.Module):
    def __init__(self,num_in_features,num_out_features,num_points):
        super().__init__()
        self.num_points = num_points
        self.bn1 = nn.BatchNorm1d(num_points)
        self.bn2 = nn.BatchNorm1d(num_points)
        self.bn3 = nn.BatchNorm1d(num_points)
        self.linear1 = nn.Sequential(nn.Linear(num_in_features, num_in_features//2),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear2 = nn.Sequential(nn.Linear(num_in_features//2, num_in_features//4),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear3 = nn.Sequential(nn.Linear(num_in_features//4, num_in_features//8),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear4 = nn.Linear(num_in_features//8,num_out_features)
    def forward(self,x,sort_dim=2):
        B,D,N = x.shape
        x = torch.sort(x,sort_dim)[0]
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        x = x.view(B,-1)
        return x

#reduces last dimension [B,D,N,K] -> [B,D,N]
class FSPOOL(nn.Module):
    def __init__(self,d,n):
        """
        d = size of third to last dimension in input
        n = size of last dimension in input (the one that is reduced)
        """
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(d,n))
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.normal_(self.weight)
    def forward(self,x):
        
        sorted_by_feature = torch.sort(x.permute(0,2,1,3),-1)[0]
        sorted_by_feature = self.weight*sorted_by_feature
        sorted_by_feature = torch.sum(sorted_by_feature,dim=-1)
        return sorted_by_feature.permute(0,2,1)

class MAXPOOL(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x.max(dim=-1, keepdim=False)[0]



class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Local_op, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = Dual_BN(out_channels)
        self.bn2 = Dual_BN(out_channels)

    def forward(self, x, flag):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6]) 
        x = x.permute(0, 1, 3, 2)   
        x = x.reshape(-1, d, s) 
        batch_size, _, N = x.size()
        x = F.relu(self.bn1(self.conv1(x),flag)) # B, D, N
        x = F.relu(self.bn2(self.conv2(x),flag)) # B, D, N
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x

class Pct_Rotation(nn.Module):
    def __init__(self, args, output_channels=40):
        super(Pct_Rotation, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = Dual_BN(64)
        self.bn2 = Dual_BN(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        self.pt_last = Point_Transformer_Last(args)

        self.conv_fuse = nn.Conv1d(1280, 1024, kernel_size=1, bias=False)
        self.bn3 = Dual_BN(1024)

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = Dual_BN(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = Dual_BN(256)
        self.dp2 = nn.Dropout(p=args.dropout)

        self.linear101 = nn.Linear(256, output_channels)
        self.linear100 = nn.Linear(256, args.angles)

    def forward(self, x,  rotation=False):
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        # B, D, N
        x = F.relu(self.bn1(self.conv1(x),rotation))
        # B, D, N
        x = F.relu(self.bn2(self.conv2(x),rotation))
        x = x.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=512, radius=0.15, nsample=32, xyz=xyz, points=x)         
        feature_0 = self.gather_local_0(new_feature,rotation)
        feature = feature_0.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=new_xyz, points=feature) 
        feature_1 = self.gather_local_1(new_feature,rotation)

        x = self.pt_last(feature_1,rotation)
        x = torch.cat([x, feature_1], dim=1)
        x = self.conv_fuse(x)
        x = F.leaky_relu(self.bn3(x,rotation), negative_slope=0.2)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = F.leaky_relu(self.bn6(self.linear1(x),rotation), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x),rotation), negative_slope=0.2)
        x = self.dp2(x)

        if rotation:
            x = self.linear100(x)
        else:
            x = self.linear101(x)

        return x, None, None

class Pct_Jigsaw(nn.Module):
    def __init__(self, args, output_channels=40):
        super(Pct_Jigsaw, self).__init__()
        self.args = args
        self.k = args.k1**3
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = Dual_BN(64)
        self.bn2 = Dual_BN(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        self.pt_last = Point_Transformer_Last(args)

        self.conv_fuse = nn.Conv1d(1280, 1024, kernel_size=1, bias=False)
        self.bn3 = Dual_BN(1024)

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = Dual_BN(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = Dual_BN(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear100 = nn.Linear(256, output_channels)

        self.conv6 = nn.Conv1d(64 + 1024, 512, 1, bias=False)
        self.conv7 = nn.Conv1d(512, 256, 1, bias=False)
        self.conv8 = nn.Conv1d(256, 128, 1, bias=False)
        self.conv9 = nn.Conv1d(128, self.k, 1, bias=False)
        self.bn101 = Dual_BN(512,eps=1e-03)
        self.bn102 = Dual_BN(256,eps=1e-03)
        self.bn103 = Dual_BN(128,eps=1e-03)

    def forward(self, x, jigsaw=False):
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        # B, D, N
        x = F.relu(self.bn1(self.conv1(x),jigsaw))
        # B, D, N
        x = F.relu(self.bn2(self.conv2(x),jigsaw))
        pointfeat = x
        x = x.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=512, radius=0.15, nsample=32, xyz=xyz, points=x)         
        feature_0 = self.gather_local_0(new_feature,jigsaw)
        feature = feature_0.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=new_xyz, points=feature) 
        feature_1 = self.gather_local_1(new_feature,jigsaw)

        x = self.pt_last(feature_1,jigsaw)
        x = torch.cat([x, feature_1], dim=1)
        x = self.conv_fuse(x)
        x = F.leaky_relu(self.bn3(x,jigsaw), negative_slope=0.2)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)

        # x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        # x = self.dp1(x)
        # x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        # x = self.dp2(x)
        # x = self.linear100(x)
        if jigsaw:
            x = x.view(-1, 1024, 1).repeat(1, 1, self.args.num_points)
            # print(x.shape,feature_1.shape)
            x = torch.cat([x, pointfeat], 1)
            x = F.relu(self.bn101(self.conv6(x),jigsaw))
            x = F.relu(self.bn102(self.conv7(x),jigsaw))
            x = F.relu(self.bn103(self.conv8(x),jigsaw))
            x = self.conv9(x)
            x = x.transpose(2,1).contiguous()
            x = F.log_softmax(x.view(-1,self.k), dim=-1)
            x = x.view(batch_size, self.args.num_points, self.k)
        else:
            x = F.leaky_relu(self.bn6(self.linear1(x),jigsaw), negative_slope=0.2)
            x = self.dp1(x)
            x = F.leaky_relu(self.bn7(self.linear2(x),jigsaw), negative_slope=0.2)
            x = self.dp2(x)
            x = self.linear100(x)

        return x, None, None


class Point_Transformer_Last(nn.Module):
    def __init__(self, args, channels=256):
        super(Point_Transformer_Last, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = Dual_BN(channels)
        self.bn2 = Dual_BN(channels)

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

    def forward(self, x, flag):
        # 
        # b, 3, npoint, nsample  
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample 
        # permute reshape
        batch_size, _, N = x.size()

        # B, D, N
        x = F.relu(self.bn1(self.conv1(x),flag))
        x = F.relu(self.bn2(self.conv2(x),flag))
        x1 = self.sa1(x,flag)
        x2 = self.sa2(x1,flag)
        x3 = self.sa3(x2,flag)
        x4 = self.sa4(x3,flag)
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
        self.after_norm = Dual_BN(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, flag):
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
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r),flag))
        x = x + x_r
        return x

class PointNet_Rotation(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet_Rotation, self).__init__()
        self.args = args
        self.stn = STN3d()
        self.fstn = STNkd(k=64)
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = Dual_BN(64,eps=1e-03)
        self.bn2 = Dual_BN(64,eps=1e-03)
        self.bn3 = Dual_BN(64,eps=1e-03)
        self.bn4 = Dual_BN(128,eps=1e-03)
        self.bn5 = Dual_BN(args.emb_dims,eps=1e-03)

        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = Dual_BN(512,eps=1e-03)
        self.dp1 = nn.Dropout(p=0.3)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn7 = Dual_BN(256,eps=1e-03)
        self.dp2 = nn.Dropout(p=0.3)
        self.linear3 = nn.Linear(256,output_channels)
        # self.pool = MLPPool(args.emb_dims,1,args.num_points)

        if args.rotation:
            self.linear4 = nn.Linear(256,args.angles)

    def forward(self, x, rotation=False):
        batch_size = x.size(0)
        trans = self.stn(x, rotation)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x),rotation))
        x = F.relu(self.bn2(self.conv2(x),rotation))
        trans_feat = self.fstn(x, rotation)
        x = x.transpose(2,1)
        x = torch.bmm(x, trans_feat)
        x = x.transpose(2,1)
        x = F.relu(self.bn3(self.conv3(x),rotation))
        x = F.relu(self.bn4(self.conv4(x),rotation))
        x = F.relu(self.bn5(self.conv5(x),rotation))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x),rotation))
        x = self.dp1(x)
        x = F.relu(self.bn7(self.linear2(x),rotation))
        x = self.dp2(x)
        if not rotation:
            x = self.linear3(x)
        else:
            x = self.linear4(x)
        
        return x, trans, trans_feat

class PointNet_Jigsaw(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet_Jigsaw, self).__init__()
        self.args = args
        self.k = (args.k1)**3
        self.stn = STN3d()
        self.fstn = STNkd(k=64)
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = Dual_BN(64,eps=1e-03)
        self.bn2 = Dual_BN(64,eps=1e-03)
        self.bn3 = Dual_BN(64,eps=1e-03)
        self.bn4 = Dual_BN(128,eps=1e-03)
        self.bn5 = Dual_BN(args.emb_dims)

        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = Dual_BN(512,eps=1e-03)
        self.dp1 = nn.Dropout(p=0.3)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn7 = Dual_BN(256,eps=1e-03)
        self.dp2 = nn.Dropout(p=0.3)
        self.linear3 = nn.Linear(256,output_channels)

        self.conv6 = nn.Conv1d(64 + args.emb_dims, 512, 1, bias=False)
        self.conv7 = nn.Conv1d(512, 256, 1, bias=False)
        self.conv8 = nn.Conv1d(256, 128, 1, bias=False)
        self.conv9 = nn.Conv1d(128, self.k, 1, bias=False)
        self.bn8 = Dual_BN(512,eps=1e-03)
        self.bn9 = Dual_BN(256,eps=1e-03)
        self.bn10 = Dual_BN(128,eps=1e-03)

    def forward(self, x, jigsaw=False):
        batchsize = x.size()[0]
        trans = self.stn(x,jigsaw)
        x = x.transpose(2, 1)
        x = torch.bmm(x,trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x),jigsaw))
        x = F.relu(self.bn2(self.conv2(x),jigsaw))
        trans_feat = self.fstn(x,jigsaw)
        x = x.transpose(2,1)
        x = torch.bmm(x, trans_feat)
        x = x.transpose(2,1)
        pointfeat = x
        x = F.relu(self.bn3(self.conv3(x),jigsaw))
        x = F.relu(self.bn4(self.conv4(x),jigsaw))
        x = F.relu(self.bn5(self.conv5(x),jigsaw))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        if not jigsaw:
            x = F.relu(self.bn6(self.linear1(x),jigsaw))
            x = self.dp1(x)
            x = F.relu(self.bn7(self.linear2(x),jigsaw))
            x = self.dp2(x)
            x = self.linear3(x)
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, self.args.num_points)
            x = torch.cat([x, pointfeat], 1)
            x = F.relu(self.bn8(self.conv6(x),jigsaw))
            x = F.relu(self.bn9(self.conv7(x),jigsaw))
            x = F.relu(self.bn10(self.conv8(x),jigsaw))
            x = self.conv9(x)
            x = x.transpose(2,1).contiguous()
            x = F.log_softmax(x.view(-1,self.k), dim=-1)
            x = x.view(batchsize, self.args.num_points, self.k)
        
        return x, trans, trans_feat

class DGCNN_Rotation(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN_Rotation, self).__init__()
        self.args = args
        self.k = args.k
        # self.FSPool_local=args.fspool_local
        # self.FSPool_global = args.fspool_global
        # self.MLPPool_global = args.mlppool_global
        self.bn1 = Dual_BN2d(64)
        self.bn2 = Dual_BN2d(64)
        self.bn3 = Dual_BN2d(128)
        self.bn4 = Dual_BN2d(256)
        self.bn5 = Dual_BN(args.emb_dims)

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

        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = Dual_BN(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = Dual_BN(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

        if args.rotation:
            # self.linear4 = nn.Linear(args.emb_dims*2, 512, bias=False)
            # self.bn8 = Dual_BN(512)
            # self.dp3 = nn.Dropout(p=args.dropout)
            # self.linear5 = nn.Linear(512, 256)
            # self.bn9 = Dual_BN(256)
            # self.dp4 = nn.Dropout(p=args.dropout)
            self.linear4 = nn.Linear(256, args.angles)
        
        # if(self.FSPool_local):
        #     self.pool1 = FSPOOL(64,args.k)
        #     self.pool2 = FSPOOL(64,args.k)
        #     self.pool3 = FSPOOL(128,args.k)
        #     self.pool4 = FSPOOL(256,args.k)
        # else:
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
        x = self.lrelu(self.bn1(self.conv1(x), rotation))
        x1 = self.pool1(x)

        x = get_graph_feature(x1, k=self.k)
        x = self.lrelu(self.bn2(self.conv2(x), rotation))
        x2 = self.pool2(x)

        x = get_graph_feature(x2, k=self.k)
        x = self.lrelu(self.bn3(self.conv3(x), rotation))
        x3 = self.pool3(x)

        x = get_graph_feature(x3, k=self.k)
        x = self.lrelu(self.bn4(self.conv4(x), rotation))
        x4 = self.pool4(x)

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.lrelu(self.bn5(self.conv5(x), rotation))

        # if(self.FSPool_global):
        #     x = x.unsqueeze(2)
        #     x1 = self.pool5(x).view(batch_size,-1)
        #     x2 = self.pool6(x).view(batch_size,-1)
        #     x = torch.cat((x1, x2), 1)
        # elif(self.MLPPool_global):
        #     x = self.pool5(x)
        # else:
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x),rotation), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x),rotation), negative_slope=0.2)
        x = self.dp2(x)
        
        if not rotation:
            x = self.linear3(x)
        else:
            x = self.linear4(x)
        return x, None, None

class DGCNN_Jigsaw(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN_Jigsaw, self).__init__()
        self.args = args
        self.k = args.k
        self.k1 = args.k1**3
        # self.FSPool_local=args.fspool_local
        # self.FSPool_global = args.fspool_global
        # self.MLPPool_global = args.mlppool_global
        self.bn1 = Dual_BN2d(64)
        self.bn2 = Dual_BN2d(64)
        self.bn3 = Dual_BN2d(128)
        self.bn4 = Dual_BN2d(256)
        self.bn5 = Dual_BN(args.emb_dims)
        

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
        
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = Dual_BN(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = Dual_BN(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

        if args.jigsaw:

            self.conv6 = nn.Conv1d(64 + args.emb_dims * 2, 512, 1, bias=False)
            self.conv7 = nn.Conv1d(512, 256, 1, bias=False)
            self.conv8 = nn.Conv1d(256, 128, 1, bias=False)
            self.conv9 = nn.Conv1d(128, self.k1, 1, bias=False)
            self.bn8 = Dual_BN(512)
            self.bn9 = Dual_BN(256)
            self.bn10 = Dual_BN(128)

        self.pool1 = MAXPOOL()
        self.pool2 = MAXPOOL()
        self.pool3 = MAXPOOL()
        self.pool4 = MAXPOOL()
        # if(self.FSPool_global):
        #     self.pool5 = FSPOOL(1024,1024)
        #     self.pool6 = FSPOOL(1024,1024)
        # elif(self.MLPPool_global):
        # self.pool5 = MLPPool(1024,2,1024)
    
    def forward(self, x, jigsaw=False):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.lrelu(self.bn1(self.conv1(x), jigsaw))
        x1 = self.pool1(x)

        x = get_graph_feature(x1, k=self.k)
        x = self.lrelu(self.bn2(self.conv2(x), jigsaw))
        x2 = self.pool2(x)

        pointfeat = x2

        x = get_graph_feature(x2, k=self.k)
        x = self.lrelu(self.bn3(self.conv3(x), jigsaw))
        x3 = self.pool3(x)

        x = get_graph_feature(x3, k=self.k)
        x = self.lrelu(self.bn4(self.conv4(x), jigsaw))
        x4 = self.pool4(x)

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.lrelu(self.bn5(self.conv5(x), jigsaw))

        # if(self.FSPool_global):
        #     x = x.unsqueeze(2)
        #     x1 = self.pool5(x).view(batch_size,-1)
        #     x2 = self.pool6(x).view(batch_size,-1)
        #     x = torch.cat((x1, x2), 1)
        # elif(self.MLPPool_global):
        #     x = self.pool5(x)
        # else:
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        if jigsaw:
            x = x.view(-1, self.args.emb_dims * 2, 1).repeat(1, 1, self.args.num_points)
            x = torch.cat([x, pointfeat], 1)
            
            x = F.leaky_relu(self.bn8(self.conv6(x),jigsaw), negative_slope=0.2)
            x = F.leaky_relu(self.bn9(self.conv7(x),jigsaw), negative_slope=0.2)
            x = F.leaky_relu(self.bn10(self.conv8(x),jigsaw), negative_slope=0.2)
            x = self.conv9(x)
            x = x.transpose(2,1).contiguous()
            x = F.log_softmax(x.view(-1,self.k1), dim=-1)
            x = x.view(batch_size, self.args.num_points, self.k1)
        else:
            x = F.leaky_relu(self.bn6(self.linear1(x),jigsaw), negative_slope=0.2)
            x = self.dp1(x)
            x = F.leaky_relu(self.bn7(self.linear2(x),jigsaw), negative_slope=0.2)
            x = self.dp2(x)
            x = self.linear3(x)
        # if not rotation:

        # else:
        #     x = F.leaky_relu(self.bn8(self.linear4(x)), negative_slope=0.2)
        #     x = self.dp3(x)
        #     x = F.leaky_relu(self.bn9(self.linear5(x)), negative_slope=0.2)
        #     x = self.dp4(x)
        #     x = self.linear6(x)
        return x, None, None