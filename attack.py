'''
Description: 
Autor: Jiachen Sun
Date: 2021-01-18 23:21:07
LastEditors: Jiachen Sun
LastEditTime: 2021-04-21 23:00:22
'''
import time
import torch
import numpy as np
import torch.nn as nn
import os
import sys
np.random.seed(666)
torch.manual_seed(666)
torch.cuda.manual_seed_all(666)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False
import torch.nn.functional as F
from util import cross_entropy_with_probs, cal_loss, margin_logit_loss, cal_loss_no_reduce, margin_logit_loss_reduce
 
def pgd_attack(model,data,labels,eps=0.01,alpha=0.0002,iters=50,repeat=1,mixup=False):
    model.eval()
    max_loss = -1
    best_examples=None
    for i in range(repeat):
        adv_data=data.clone()
        adv_data=adv_data+(torch.rand_like(adv_data)*eps*2-eps)
        # adv_data = torch.clamp(data,-1,1)
        adv_data.detach()
        for i in range(iters):
            adv_data.requires_grad=True
            outputs,_,trans = model(adv_data)
            if mixup:
                loss = cross_entropy_with_probs(outputs,labels)
            else:
                loss = cal_loss(outputs,None,labels)
            # print(torch.autograd.grad(loss,adv_data,create_graph=True))   
            loss.backward()
            with torch.no_grad():
                adv_data = adv_data + alpha*adv_data.grad.sign()
                delta = adv_data-data
                delta = torch.clamp(delta,-eps,eps)
                adv_data = data+delta
               #If points outside the unit cube are invalid then
                # adv_data = torch.clamp(adv_data,-1,1)
            if loss > max_loss:
                max_loss=loss
                best_examples=adv_data

        outputs,_,trans = model(best_examples)
        if mixup:
            loss = cross_entropy_with_probs(outputs,labels)
        else:
            loss = cal_loss(outputs,None,labels)
        if loss > max_loss:
            max_loss=loss
            best_examples=adv_data.cpu()
            
    return best_examples.cuda()

def pgd_attack_ensemble(model1,model2,model3,data,labels,eps=0.01,alpha=0.0002,iters=50,repeat=1,mixup=False):
    model1.eval()
    model2.eval()
    model3.eval()
    max_loss = -1
    best_examples=None
    for i in range(repeat):
        adv_data=data.clone()
        adv_data=adv_data+(torch.rand_like(adv_data)*eps*2-eps)
        # adv_data = torch.clamp(data,-1,1)
        adv_data.detach()
        for i in range(iters):
            adv_data.requires_grad=True
            logits1,_,_ = model1(adv_data)
            logits2,_,_ = model2(adv_data)
            logits3,_,_ = model3(adv_data)
            logits = logits1 + logits2 + logits3
            # logits = torch.stack([logits1,logits2,logits3])
            # logits = torch.max(logits,dim=0)[0]
            if mixup:
                raise NotImplementedError('not implemented')
            else:
                loss = cal_loss(logits,None,labels)

            # print(torch.autograd.grad(loss,adv_data,create_graph=True))   
            loss.backward()
            with torch.no_grad():
                adv_data = adv_data + alpha*adv_data.grad.sign()
                delta = adv_data-data
                delta = torch.clamp(delta,-eps,eps)
                adv_data = data+delta
               #If points outside the unit cube are invalid then
                # adv_data = torch.clamp(adv_data,-1,1)
            if loss > max_loss:
                max_loss=loss
                best_examples=adv_data

        logits1,_,_ = model1(adv_data)
        logits2,_,_ = model2(adv_data)
        logits3,_,_ = model3(adv_data)
        # logits = torch.stack([logits1,logits2,logits3])
        # logits = torch.max(logits,dim=0)[0]
        logits = logits1 + logits2 + logits3
        if mixup:
            raise NotImplementedError('not implemented')
        else:
            loss = cal_loss(logits,None,labels)
        if loss > max_loss:
            max_loss=loss
            best_examples=adv_data.cpu()
            
    return best_examples.cuda()
 
def pgd_attack_feature(model,data,labels,eps=0.01,alpha=0.0002,iters=50,repeat=1,mixup=False):
    model.eval()
    max_loss = -1
    best_examples=None
    for i in range(repeat):
        adv_data=data.clone()
        adv_data=adv_data+(torch.rand_like(adv_data)*eps*2-eps)
        # adv_data = torch.clamp(data,-1,1)
        adv_data.detach()
        for i in range(iters):
            adv_data.requires_grad=True
            _,outputs,trans = model(adv_data)
            if mixup:
                loss = cross_entropy_with_probs(outputs,labels)
            else:
                loss = torch.mean(torch.abs(outputs - labels))
            # print(torch.autograd.grad(loss,adv_data,create_graph=True))   
            loss.backward(retain_graph=True)
            with torch.no_grad():
                adv_data = adv_data + alpha*adv_data.grad.sign()
                delta = adv_data-data
                delta = torch.clamp(delta,-eps,eps)
                adv_data = data+delta
               #If points outside the unit cube are invalid then
                # adv_data = torch.clamp(adv_data,-1,1)
            if loss > max_loss:
                max_loss=loss
                best_examples=adv_data

        _,outputs,trans = model(best_examples)
        if mixup:
            loss = cross_entropy_with_probs(outputs,labels)
        else:
            loss = torch.mean(torch.abs(outputs - labels))
        if loss > max_loss:
            max_loss=loss
            best_examples=adv_data.cpu()
            
    return best_examples.cuda()

def pgd_attack_margin(model,data,labels,eps=0.01,alpha=0.0002,iters=50,repeat=1,mixup=False):
    model.eval()
    max_loss = -1e5
    best_examples=None
    for i in range(repeat):
        adv_data=data.clone()
        adv_data=adv_data+(torch.rand_like(adv_data)*eps*2-eps)
        # adv_data = torch.clamp(data,-1,1)
        adv_data.detach()
        for i in range(iters):
            adv_data.requires_grad=True
            outputs,_,trans = model(adv_data)
            if mixup:
                loss = cross_entropy_with_probs(outputs,labels)
            else:
                loss = margin_logit_loss_reduce(outputs,None,labels)
            # print(torch.autograd.grad(loss,adv_data,create_graph=True))   
            loss.backward()
            with torch.no_grad():
                adv_data = adv_data + alpha*adv_data.grad.sign()
                delta = adv_data-data
                delta = torch.clamp(delta,-eps,eps)
                adv_data = data+delta
               #If points outside the unit cube are invalid then
                # adv_data = torch.clamp(adv_data,-1,1)
            if loss > max_loss:
                max_loss=loss
                best_examples=adv_data

        outputs,_,trans = model(best_examples)
        if mixup:
            loss = cross_entropy_with_probs(outputs,labels)
        else:
            loss = margin_logit_loss_reduce(outputs,None,labels)
        if loss > max_loss:
            max_loss=loss
            best_examples=adv_data.cpu()
            
    return best_examples.cuda()

def mim_margin(model,data,labels,eps=0.01,alpha=0.0002,iters=50,repeat=1,mixup=False):
    model.eval()
    max_loss = -1e5
    best_examples=None
    for i in range(repeat):
        adv_data=data.clone()
        adv_data=adv_data+(torch.rand_like(adv_data)*eps*2-eps)
        # adv_data = torch.clamp(data,-1,1)
        adv_data.detach()
        g = 0
        for i in range(iters):
            adv_data.requires_grad=True
            outputs,_,trans = model(adv_data)
            if mixup:
                loss = cross_entropy_with_probs(outputs,labels)
            else:
                loss = margin_logit_loss_reduce(outputs,None,labels)
            # print(torch.autograd.grad(loss,adv_data,create_graph=True))   
            loss.backward()
            with torch.no_grad():
                g += adv_data.grad / torch.max(torch.abs(adv_data.grad))
                adv_data = adv_data + alpha*g
                delta = adv_data-data
                delta = torch.clamp(delta,-eps,eps)
                adv_data = data+delta
               #If points outside the unit cube are invalid then
                # adv_data = torch.clamp(adv_data,-1,1)
            if loss > max_loss:
                max_loss=loss
                best_examples=adv_data

        outputs,_,trans = model(best_examples)
        if mixup:
            loss = cross_entropy_with_probs(outputs,labels)
        else:
            loss = margin_logit_loss_reduce(outputs,None,labels)
        if loss > max_loss:
            max_loss=loss
            best_examples=adv_data.cpu()
            
    return best_examples.cuda()

def mim(model,data,labels,eps=0.01,alpha=0.0002,iters=50,repeat=1,mixup=False):
    model.eval()
    max_loss = -1e5
    best_examples=None
    for i in range(repeat):
        adv_data=data.clone()
        adv_data=adv_data+(torch.rand_like(adv_data)*eps*2-eps)
        # adv_data = torch.clamp(data,-1,1)
        adv_data.detach()
        g = 0
        for i in range(iters):
            adv_data.requires_grad=True
            outputs,_,trans = model(adv_data)
            if mixup:
                loss = cross_entropy_with_probs(outputs,labels)
            else:
                loss = cal_loss(outputs,None,labels)
            # print(torch.autograd.grad(loss,adv_data,create_graph=True))   
            loss.backward()
            with torch.no_grad():
                g += adv_data.grad / torch.max(torch.abs(adv_data.grad))
                adv_data = adv_data + alpha*g
                delta = adv_data-data
                delta = torch.clamp(delta,-eps,eps)
                adv_data = data+delta
               #If points outside the unit cube are invalid then
                # adv_data = torch.clamp(adv_data,-1,1)
            if loss > max_loss:
                max_loss=loss
                best_examples=adv_data

        outputs,_,trans = model(best_examples)
        if mixup:
            loss = cross_entropy_with_probs(outputs,labels)
        else:
            loss = cal_loss(outputs,None,labels)
        if loss > max_loss:
            max_loss=loss
            best_examples=adv_data.cpu()
            
    return best_examples.cuda()



def bim(model,data,labels,eps=0.01,alpha=0.0002,iters=50,repeat=1,mixup=False):
    model.eval()
    max_loss = -1e5
    best_examples=None
    for i in range(repeat):
        adv_data=data.clone()
        # adv_data=adv_data+(torch.rand_like(adv_data)*eps*2-eps)
        # adv_data = torch.clamp(data,-1,1)
        adv_data.detach()
        for i in range(iters):
            adv_data.requires_grad=True
            outputs,_,trans = model(adv_data)
            if mixup:
                loss = cross_entropy_with_probs(outputs,labels)
            else:
                loss = margin_logit_loss_reduce(outputs,None,labels)
            # print(torch.autograd.grad(loss,adv_data,create_graph=True))   
            loss.backward()
            with torch.no_grad():
                adv_data = adv_data + alpha*adv_data.grad.sign()
                delta = adv_data-data
                delta = torch.clamp(delta,-eps,eps)
                adv_data = data+delta
               #If points outside the unit cube are invalid then
                # adv_data = torch.clamp(adv_data,-1,1)
            if loss > max_loss:
                max_loss=loss
                best_examples=adv_data

        outputs,_,trans = model(best_examples)
        if mixup:
            loss = cross_entropy_with_probs(outputs,labels)
        else:
            loss = margin_logit_loss_reduce(outputs,None,labels)
        if loss > max_loss:
            max_loss=loss
            best_examples=adv_data.cpu()
            
    return best_examples.cuda()

def pgd_attack_seg(model,data,labels,number,eps=0.01,alpha=0.0002,iters=50,repeat=1):
    model.eval()
    max_loss = -1e5
    best_examples=None
    for i in range(repeat):
        adv_data=data.clone()
        adv_data=adv_data+(torch.rand_like(adv_data)*eps*2-eps)
        # adv_data = torch.clamp(data,-1,1)
        adv_data.detach()
        for i in range(iters):
            adv_data.requires_grad=True
            logits,_,_ = model(adv_data)
            logits = logits.view(-1,number)
            labels = labels.view(-1,1)[:,0]
            loss= F.nll_loss(logits,labels)
            # loss = cal_loss(outputs,None,labels)
            # print(torch.autograd.grad(loss,adv_data,create_graph=True))   
            loss.backward()
            with torch.no_grad():
                adv_data = adv_data + alpha*adv_data.grad.sign()
                delta = adv_data-data
                delta = torch.clamp(delta,-eps,eps)
                adv_data = data+delta
               #If points outside the unit cube are invalid then
                # adv_data = torch.clamp(adv_data,-1,1)
            if loss > max_loss:
                max_loss=loss
                best_examples=adv_data.cpu()

        logits,_,_ = model(adv_data)
        logits = logits.view(-1,number)
        labels = labels.view(-1,1)[:,0]
        loss= F.nll_loss(logits,labels)
        if loss > max_loss:
            max_loss=loss
            best_examples=adv_data.cpu()
            
    return best_examples.cuda()

def pgd_attack_seg_feature(model,data,labels,number,eps=0.01,alpha=0.0002,iters=50,repeat=1):
    model.eval()
    max_loss = -1e5
    best_examples=None
    for i in range(repeat):
        adv_data=data.clone()
        adv_data=adv_data+(torch.rand_like(adv_data)*eps*2-eps)
        # adv_data = torch.clamp(data,-1,1)
        adv_data.detach()
        for i in range(iters):
            adv_data.requires_grad=True
            logits,outputs,_ = model(adv_data)
            # logits = logits.view(-1,number)
            # labels = labels.view(-1,1)[:,0]
            # loss= F.nll_loss(logits,labels)
            loss = torch.mean(torch.abs(outputs - labels))
            
            # loss = cal_loss(outputs,None,labels)
            # print(torch.autograd.grad(loss,adv_data,create_graph=True))   
            loss.backward(retain_graph=True)
            with torch.no_grad():
                adv_data = adv_data + alpha*adv_data.grad.sign()
                delta = adv_data-data
                delta = torch.clamp(delta,-eps,eps)
                adv_data = data+delta
               #If points outside the unit cube are invalid then
                # adv_data = torch.clamp(adv_data,-1,1)
            if loss > max_loss:
                max_loss=loss
                best_examples=adv_data.cpu()

        logits,outputs,_ = model(adv_data)
        # logits = logits.view(-1,number)
        # labels = labels.view(-1,1)[:,0]
        # loss= F.nll_loss(logits,labels)
        loss = torch.mean(torch.abs(outputs - labels))
        if loss > max_loss:
            max_loss=loss
            best_examples=adv_data.cpu()
            
    return best_examples.cuda()

def pgd_attack_partseg(model,data,labels,one_hot,number,eps=0.01,alpha=0.0002,iters=50,repeat=1):
    model.eval()
    max_loss = -1e5
    best_examples=None
    for i in range(repeat):
        adv_data=data.clone()
        adv_data=adv_data+(torch.rand_like(adv_data)*eps*2-eps)
        # adv_data = torch.clamp(data,-1,1)
        adv_data.detach()
        for i in range(iters):
            adv_data.requires_grad=True
            seg_pred = model(adv_data, one_hot)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            loss = cal_loss(seg_pred.view(-1, number),None, labels.view(-1,1).squeeze())
            # loss = cal_loss(outputs,None,labels)
            # print(torch.autograd.grad(loss,adv_data,create_graph=True))   
            loss.backward()
            with torch.no_grad():
                adv_data = adv_data + alpha*adv_data.grad.sign()
                delta = adv_data-data
                delta = torch.clamp(delta,-eps,eps)
                adv_data = data+delta
               #If points outside the unit cube are invalid then
                # adv_data = torch.clamp(adv_data,-1,1)
            if loss > max_loss:
                max_loss=loss
                best_examples=adv_data.cpu()

        seg_pred = model(adv_data, one_hot)
        seg_pred = seg_pred.permute(0, 2, 1).contiguous()
        loss = cal_loss(seg_pred.view(-1, number),None, labels.view(-1,1).squeeze())
        
        if loss > max_loss:
            max_loss=loss
            best_examples=adv_data.cpu()
            
    return best_examples.cuda()

def spsa(model,data,labels_og,eps=0.01,alpha=0.001,iters=2000,samples=32):
    model.eval()
    final_adv = []
    with torch.no_grad():
        BATCH_SIZE = data.shape[0]
        for b in range(BATCH_SIZE):
            adv_data=torch.squeeze(data[b].clone())
            labels = torch.ones_like(labels_og) * labels_og[b]
            adv_data.detach()
            adv_data_og =  adv_data.clone()
            adv_data=adv_data+(torch.rand_like(adv_data)*eps*2-eps)
            for _ in range(iters):
                est_g = torch.zeros_like(adv_data)
                for j in range(samples // BATCH_SIZE):
                    adv_data_repeat = adv_data.repeat([BATCH_SIZE,1,1])
                    pert = torch.rand_like(adv_data_repeat) - 0.5
                    adv_data_repeat_1 = adv_data_repeat + pert.sign() * eps
                    logits_1,_,_ = model(adv_data_repeat_1)
                    loss_1 = cal_loss_no_reduce(logits_1,labels)

                    adv_data_repeat_2 = adv_data_repeat - pert.sign() * eps
                    logits_2,_,_ = model(adv_data_repeat_2)
                    loss_2 = cal_loss_no_reduce(logits_2,labels)

                    sub_loss = torch.reshape(loss_1 - loss_2, [-1,1,1]).repeat([1,adv_data_repeat.shape[1],adv_data_repeat.shape[2]])
                    est_g += torch.sum(sub_loss / (2 * eps * pert.sign()),0)
                
                est_g = est_g / samples
                adv_data = adv_data + alpha * est_g.sign()
                delta = adv_data-adv_data_og
                delta = torch.clamp(delta,-eps,eps)
                adv_data = adv_data_og+delta
            final_adv.append(adv_data)

        final_adv = torch.stack(final_adv)
    
    return final_adv.cuda()

def nattack(model,data,labels_og,eps=0.01,alpha=0.001,iters=2000,variance=0.001,samples=32):
    model.eval()
    final_adv = []
    with torch.no_grad():
        BATCH_SIZE = data.shape[0]
        for b in range(BATCH_SIZE):
            adv_data=torch.squeeze(data[b].clone())
            labels = torch.ones_like(labels_og) * labels_og[b]
            adv_data.detach()
            adv_data_og =  adv_data.clone()
            adv_data=adv_data+(torch.rand_like(adv_data)*eps*2-eps)
            mu = torch.zeros_like(adv_data_og)
            
            for _ in range(iters):
                est_g = torch.zeros_like(adv_data)
                loss_all = []
                loss_sum = 0
                perts_sum = torch.zeros_like(adv_data)
                losses_perts_sum = 0

                for j in range(samples // BATCH_SIZE):
                    adv_data_repeat = adv_data.repeat([BATCH_SIZE,1,1])
                    pert = torch.normal(0.0,1.0,size=adv_data_repeat.shape).cuda()
                    # print(pert.shape)
                    mu_perts = mu.repeat([BATCH_SIZE,1,1]) + pert * variance

                    arctanh_x = torch.atanh(adv_data_repeat)
                    delta = torch.tanh(arctanh_x + mu_perts) - adv_data_repeat
                    delta = torch.clamp(delta,-eps,eps)
                    adv_data_repeat_1 = adv_data_repeat + delta
                    logits,_,_ = model(adv_data_repeat_1)
                    loss = cal_loss_no_reduce(logits,labels)
                    loss_sum += torch.sum(loss,0)
                    loss_all.append(loss)
                    perts_sum += torch.sum(pert,0)
                    losses_perts_sum += torch.sum(torch.reshape(loss, (-1,1,1)) * pert, 0)

                    # est_g += losses_perts_mean - torch.mean(loss)
                loss_all = torch.cat(loss_all)
                est_g = ( - (loss_sum / samples) * (perts_sum / samples) + (losses_perts_sum / samples)) / ((torch.std(loss_all)+1e-7) * variance)
                # print((losses_perts_sum / samples).shape)
                mu = mu + alpha * est_g.sign()
                delta = torch.tanh(torch.atanh(adv_data) + mu)-adv_data_og
                delta = torch.clamp(delta,-eps,eps)
                adv_data = adv_data_og+delta
                # print(adv_data.shape)
                
            final_adv.append(adv_data)

        final_adv = torch.stack(final_adv)
    
    return final_adv.cuda()

def nes(model,data,labels_og,eps=0.01,alpha=0.001,iters=2000,variance=0.1,samples=32):

    model.eval()
    final_adv = []
    with torch.no_grad():
        BATCH_SIZE = data.shape[0]
        for b in range(BATCH_SIZE):
            adv_data=torch.squeeze(data[b].clone())
            labels = torch.ones_like(labels_og) * labels_og[b]
            adv_data.detach()
            adv_data_og =  adv_data.clone()
            adv_data=adv_data+(torch.rand_like(adv_data)*eps*2-eps)
            for _ in range(iters):
                est_g = torch.zeros_like(adv_data)
                for j in range(samples // BATCH_SIZE):
                    adv_data_repeat = adv_data.repeat([BATCH_SIZE,1,1])
                    # print(adv_data_repeat.shape)
                    pert = torch.normal(0.0,1.0,size=adv_data_repeat.shape).cuda()
                    
                    adv_data_repeat_1 = adv_data_repeat + pert * variance
                    logits_1,_,_ = model(adv_data_repeat_1)
                    loss_1 = cal_loss_no_reduce(logits_1,labels)

                    adv_data_repeat_2 = adv_data_repeat - pert * variance
                    logits_2,_,_ = model(adv_data_repeat_2)
                    loss_2 = cal_loss_no_reduce(logits_2,labels)
                    # print(loss_2.shape)

                    sub_loss = torch.reshape(loss_1 - loss_2, [-1,1,1]).repeat([1,adv_data_repeat.shape[1],adv_data_repeat.shape[2]])
                    # print(sub_loss.shape)
                    est_g += torch.sum(sub_loss * pert / (2 * variance),0)
                
                est_g = est_g / samples
                adv_data = adv_data + alpha * est_g.sign()
                delta = adv_data-adv_data_og
                delta = torch.clamp(delta,-eps,eps)
                adv_data = adv_data_og+delta
            final_adv.append(adv_data)

        final_adv = torch.stack(final_adv)
    
    return final_adv.cuda()


def evolution(model,data,labels_og,eps=0.01,iters=2000,variance=0.05,samples=32,k=8):

    model.eval()
    final_adv = []
    with torch.no_grad():
        BATCH_SIZE = data.shape[0]
        for b in range(BATCH_SIZE):
            adv_data=torch.squeeze(data[b].clone())
            labels = torch.ones_like(labels_og) * labels_og[b]
            adv_data.detach()
            adv_data_og =  adv_data.clone()

            pert = torch.rand(samples,adv_data.shape[0],adv_data.shape[1])*eps*2-eps
            pert = pert.cuda()
            for _ in range(iters):
                
                loss_iter = []
                for j in range(samples // BATCH_SIZE):
                    adv_data_repeat = adv_data_og.repeat([BATCH_SIZE,1,1])
                    pert[j*BATCH_SIZE:(j+1)*BATCH_SIZE] += torch.normal(0.0,variance,size=adv_data_repeat.shape).cuda()
                    
                    pert[j*BATCH_SIZE:(j+1)*BATCH_SIZE] = torch.clamp(adv_data_repeat + pert[j*BATCH_SIZE:(j+1)*BATCH_SIZE] - adv_data_og,-eps,eps)
                    adv_data_repeat = adv_data_repeat + pert[j*BATCH_SIZE:(j+1)*BATCH_SIZE]
                    
                    logits,_,_ = model(adv_data_repeat)
                    loss = cal_loss_no_reduce(logits,labels)
                    loss_iter.append(loss)
                
                loss_iter = torch.cat(loss_iter)
                _,index = torch.topk(loss_iter, k, largest = True, sorted = True)

                # print(pert.shape)
                # print(index)
                pert = pert[index].repeat([samples // k,1,1])

            # index = torch.argmax(loss_iter)
            adv_data = pert[0] + adv_data_og

            final_adv.append(adv_data)

        final_adv = torch.stack(final_adv)
    
    return final_adv.cuda()



class APGDAttack():
    def __init__(self, model, n_iter=100, norm='Linf', n_restarts=1, eps=None,
                 seed=0, loss='ce', eot_iter=1, rho=.75, verbose=False,
                 device='cuda'):
        self.model = model
        self.n_iter = n_iter
        self.eps = eps
        self.norm = norm
        self.n_restarts = n_restarts
        self.seed = seed
        self.loss = loss
        self.eot_iter = eot_iter
        self.thr_decr = rho
        self.verbose = verbose
        self.device = device
    
    def check_oscillation(self, x, j, k, y5, k3=0.75):
        t = np.zeros(x.shape[1])
        for counter5 in range(k):
          t += x[j - counter5] > x[j - counter5 - 1]
          
        return t <= k*k3*np.ones(t.shape)
        
    def check_shape(self, x):
        return x if len(x.shape) > 0 else np.expand_dims(x, 0)
    
    def dlr_loss(self, x, y):
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()
        
        return -(x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)
    
    def attack_single_run(self, x_in, y_in):
        x = x_in.clone() if len(x_in.shape) == 3 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)
        
        self.n_iter_2, self.n_iter_min, self.size_decr = max(int(0.22 * self.n_iter), 1), max(int(0.06 * self.n_iter), 1), max(int(0.03 * self.n_iter), 1)
        if self.verbose:
            print('parameters: ', self.n_iter, self.n_iter_2, self.n_iter_min, self.size_decr)
        
        if self.norm == 'Linf':
            t = 2 * torch.rand(x.shape).to(self.device).detach() - 1
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1]).to(self.device).detach() * t / (t.reshape([t.shape[0], -1]).abs().max(dim=1, keepdim=True)[0].reshape([-1, 1, 1]))
        elif self.norm == 'L2':
            t = torch.randn(x.shape).to(self.device).detach()
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1]).to(self.device).detach() * t / ((t ** 2).sum(dim=(1, 2), keepdim=True).sqrt() + 1e-12)
        # x_adv = x_adv.clamp(0., 1.)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_steps = torch.zeros([self.n_iter, x.shape[0]])
        loss_best_steps = torch.zeros([self.n_iter + 1, x.shape[0]])
        acc_steps = torch.zeros_like(loss_best_steps)
        
        if self.loss == 'ce':
            criterion_indiv = nn.CrossEntropyLoss(reduce=False, reduction='none')
        elif self.loss == 'dlr':
            criterion_indiv = self.dlr_loss
        elif self.loss == 'ce_margin':
            criterion_indiv = margin_logit_loss
        else:
            raise ValueError('unknowkn loss')
        
        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        for _ in range(self.eot_iter):
            with torch.enable_grad():
                logits,_,_ = self.model(x_adv) # 1 forward pass (eot_iter = 1)
                loss_indiv = criterion_indiv(logits, y)
                loss = loss_indiv.sum()
                    
            grad += torch.autograd.grad(loss, [x_adv])[0].detach() # 1 backward pass (eot_iter = 1)
            
        grad /= float(self.eot_iter)
        grad_best = grad.clone()
        
        acc = logits.detach().max(1)[1] == y
        acc_steps[0] = acc + 0
        loss_best = loss_indiv.detach().clone()
        
        step_size = self.eps * torch.ones([x.shape[0], 1, 1]).to(self.device).detach() * torch.Tensor([2.0]).to(self.device).detach().reshape([1, 1, 1])
        x_adv_old = x_adv.clone()
        counter = 0
        k = self.n_iter_2 + 0
        u = np.arange(x.shape[0])
        counter3 = 0
        
        loss_best_last_check = loss_best.clone()
        reduced_last_check = np.zeros(loss_best.shape) == np.zeros(loss_best.shape)
        n_reduced = 0
        
        for i in range(self.n_iter):
            ### gradient step
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()
                
                a = 0.75 if i > 0 else 1.0
                
                if self.norm == 'Linf':
                    x_adv_1 = x_adv + step_size * torch.sign(grad)
                    x_adv_1 = torch.min(torch.max(x_adv_1, x - self.eps), x + self.eps)
                    x_adv_1 = torch.min(torch.max(x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a), x - self.eps), x + self.eps)
                    
                elif self.norm == 'L2':
                    raise ValueError('not implemented yet')
                    
                x_adv = x_adv_1 + 0.
            
            ### get gradient
            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            for _ in range(self.eot_iter):
                with torch.enable_grad():
                    logits,_,_ = self.model(x_adv) # 1 forward pass (eot_iter = 1)
                    loss_indiv = criterion_indiv(logits, y)
                    loss = loss_indiv.sum()
                
                grad += torch.autograd.grad(loss, [x_adv])[0].detach() # 1 backward pass (eot_iter = 1)
                
            
            grad /= float(self.eot_iter)
            
            pred = logits.detach().max(1)[1] == y
            acc = torch.min(acc, pred)
            acc_steps[i + 1] = acc + 0
            x_best_adv[(pred == 0).nonzero().squeeze()] = x_adv[(pred == 0).nonzero().squeeze()] + 0.
            if self.verbose:
                print('iteration: {} - Best loss: {:.6f}'.format(i, loss_best.sum()))
            
            ### check step size
            with torch.no_grad():
              y1 = loss_indiv.detach().clone()
              loss_steps[i] = y1.cpu() + 0
              ind = (y1 > loss_best).nonzero().squeeze()
              x_best[ind] = x_adv[ind].clone()
              grad_best[ind] = grad[ind].clone()
              loss_best[ind] = y1[ind] + 0
              loss_best_steps[i + 1] = loss_best + 0
              
              counter3 += 1
          
              if counter3 == k:
                  fl_oscillation = self.check_oscillation(loss_steps.detach().cpu().numpy(), i, k, loss_best.detach().cpu().numpy(), k3=self.thr_decr)
                  fl_reduce_no_impr = (~reduced_last_check) * (loss_best_last_check.cpu().numpy() >= loss_best.cpu().numpy())
                  fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
                  reduced_last_check = np.copy(fl_oscillation)
                  loss_best_last_check = loss_best.clone()
                  
                  if np.sum(fl_oscillation) > 0:
                      step_size[u[fl_oscillation]] /= 2.0
                      n_reduced = fl_oscillation.astype(float).sum()
                      
                      fl_oscillation = np.where(fl_oscillation)
                      
                      x_adv[fl_oscillation] = x_best[fl_oscillation].clone()
                      grad[fl_oscillation] = grad_best[fl_oscillation].clone()
                      
                  counter3 = 0
                  k = np.maximum(k - self.size_decr, self.n_iter_min)
              
        return x_best, acc, loss_best, x_best_adv
    
    def perturb(self, x_in, y_in, best_loss=False, cheap=True):
        assert self.norm in ['Linf', 'L2']
        x = x_in.clone() if len(x_in.shape) == 3 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)
        
        adv = x.clone()
        acc = self.model(x)[0].max(1)[1] == y
        loss = -1e10 * torch.ones_like(acc).float()
        if self.verbose:
            print('-------------------------- running {}-attack with epsilon {:.4f} --------------------------'.format(self.norm, self.eps))
            print('initial accuracy: {:.2%}'.format(acc.float().mean()))
        startt = time.time()
        
        if not best_loss:
            torch.random.manual_seed(self.seed)
            torch.cuda.random.manual_seed(self.seed)
            
            if not cheap:
                raise ValueError('not implemented yet')
            
            else:
                for counter in range(self.n_restarts):
                    ind_to_fool = acc.nonzero().squeeze()
                    if len(ind_to_fool.shape) == 0: ind_to_fool = ind_to_fool.unsqueeze(0)
                    if ind_to_fool.numel() != 0:
                        x_to_fool, y_to_fool = x[ind_to_fool].clone(), y[ind_to_fool].clone()
                        best_curr, acc_curr, loss_curr, adv_curr = self.attack_single_run(x_to_fool, y_to_fool)
                        ind_curr = (acc_curr == 0).nonzero().squeeze()
                        acc[ind_to_fool[ind_curr]] = 0
                        adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                        if self.verbose:
                            print('restart {} - robust accuracy: {:.2%} - cum. time: {:.1f} s'.format(
                                counter, acc.float().mean(), time.time() - startt))
            
            return acc, adv
        
        else:
            adv_best = x.detach().clone()
            loss_best = torch.ones([x.shape[0]]).to(self.device) * (-float('inf'))
            for counter in range(self.n_restarts):
                best_curr, _, loss_curr, _ = self.attack_single_run(x, y)
                ind_curr = (loss_curr > loss_best).nonzero().squeeze()
                adv_best[ind_curr] = best_curr[ind_curr] + 0.
                loss_best[ind_curr] = loss_curr[ind_curr] + 0.
            
                if self.verbose:
                    print('restart {} - loss: {:.5f}'.format(counter, loss_best.sum()))
            
            return loss_best, adv_best


def uniform_attack(model,data,eps):
    model.eval()

    adv_data=data.clone()
    adv_data=adv_data+(torch.rand_like(adv_data)*eps*2-eps)
    # adv_data = torch.clamp(data,-1,1)
    adv_data.detach()

    return adv_data.cuda()

def gaussian_attack(model,data,eps):
    model.eval()

    adv_data=data.clone()
    pert = torch.normal(0.0,1.0,size=adv_data.shape).cuda()
    pert=torch.clamp(pert,-eps,eps)
    adv_data += pert 
    # adv_data = torch.clamp(data,-1,1)
    adv_data.detach()

    return adv_data.cuda()

def saliency(model,data,labels,number,iters):
    model.eval()

    adv_data=data.clone()
    alpha = number // iters

    for i in range(iters):
        adv_data.requires_grad=True
        outputs,_,trans = model(adv_data)
        loss = cal_loss(outputs,None,labels)
        loss.backward()
        with torch.no_grad():
            sphere_core,_ = torch.median(adv_data, dim=2, keepdim=True)
            sphere_r = torch.sqrt(torch.sum(torch.square(adv_data - sphere_core), dim=1))
            sphere_axis = adv_data - sphere_core

            sphere_map = - torch.multiply(torch.sum(torch.multiply(adv_data.grad, sphere_axis), dim=1), torch.pow(sphere_r, 2))
            _,indice = torch.topk(sphere_map, k=adv_data.shape[2] - alpha, dim=-1, largest=False)
            tmp = torch.zeros((adv_data.shape[0], 3, adv_data.shape[2] - alpha))
            for i in range(adv_data.shape[0]):
                tmp[i] = adv_data[i][:,indice[i]]
            adv_data = tmp.clone()
            
    return adv_data.cuda()

def random_drop(model,data,number):
    model.eval()
    adv_data=data.clone()
    indices = torch.Tensor(np.random.choice(adv_data.shape[2], (adv_data.shape[0],adv_data.shape[2]-number))).long()
    tmp = torch.zeros((adv_data.shape[0], 3, adv_data.shape[2] - number))
    for i in range(adv_data.shape[0]):
        tmp[i] = adv_data[i][:,indices[i]]
    adv_data = tmp.clone()

    return adv_data.cuda()


def pgd_adding_attack(model,data,labels,number,eps=0.01,alpha=0.0002,iters=50,repeat=1,mixup=False):
    model.eval()
    max_loss = -1
    best_examples=None
    for i in range(repeat):
        indices = torch.Tensor(np.random.choice(data.shape[2], number)).long()
        adv_data_og = data.clone()[:,:,indices]
        adv_data = adv_data_og+(torch.rand_like(adv_data_og)*eps*2-eps)
        # adv_data = torch.clamp(data,-1,1)
        adv_data.detach()
        for i in range(iters):
            adv_data.requires_grad=True
            input_data = torch.cat([data,adv_data],dim=-1)
            outputs,_,trans = model(input_data)
            if mixup:
                loss = cross_entropy_with_probs(outputs,labels)
            else:
                loss = cal_loss(outputs,None,labels)
            # print(torch.autograd.grad(loss,adv_data,create_graph=True))   
            loss.backward()
            with torch.no_grad():
                adv_data = adv_data + alpha*adv_data.grad.sign()
                delta = adv_data-adv_data_og
                delta = torch.clamp(delta,-eps,eps)
                adv_data = adv_data_og+delta
               #If points outside the unit cube are invalid then
                # adv_data = torch.clamp(adv_data,-1,1)
            if loss > max_loss:
                max_loss=loss
                best_examples=adv_data

        # outputs,_,trans = model(best_examples)
        # if mixup:
        #     loss = cross_entropy_with_probs(outputs,labels)
        # else:
        #     loss = cal_loss(outputs,None,labels)
        # if loss > max_loss:
        #     max_loss=loss
        #     best_examples=adv_data.cpu()
        best_examples_f = torch.cat([data,best_examples],dim=-1)
            
    return best_examples_f.cuda()