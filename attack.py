'''
Description: 
Autor: Jiachen Sun
Date: 2021-01-18 23:21:07
LastEditors: Jiachen Sun
LastEditTime: 2021-03-31 22:32:55
'''

import torch
import numpy as np
np.random.seed(666)
torch.manual_seed(666)
torch.cuda.manual_seed_all(666)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False
import torch.nn.functional as F
from util import cross_entropy_with_probs, cal_loss
 
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
 

def pgd_attack_seg(model,data,labels,number,eps=0.01,alpha=0.0002,iters=50,repeat=1):
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
            for _ in range(iters):
                est_g = torch.zeros_like(adv_data)
                for j in range(samples // BATCH_SIZE):
                    adv_data_repeat = adv_data.repeat([BATCH_SIZE,1,1])
                    pert = torch.rand_like(adv_data_repeat)*2 - 1
                    adv_data_repeat_1 = adv_data_repeat + pert * eps
                    logits_1,_,_ = model(adv_data_repeat_1)
                    # print(logits_1.shape)
                    loss_1 = cal_loss(logits_1,None,labels)

                    adv_data_repeat_2 = adv_data_repeat - pert * eps
                    logits_2,_,_ = model(adv_data_repeat_2)
                    loss_2 = cal_loss(logits_2,None,labels)

                    est_g += torch.sum((loss_1 - loss_2) * pert / (2 * eps),0)
                
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
                    loss = cal_loss(logits,None,labels)
                    loss_sum += torch.sum(loss,0)
                    loss_all.append(loss)
                    perts_sum += torch.sum(pert,0)
                    losses_perts_sum += torch.sum(torch.reshape(loss, (-1,1,1)) * pert, 0)

                    # est_g += losses_perts_mean - torch.mean(loss)
                loss_all = torch.stack(loss_all)
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

def nes(model,data,labels_og,eps=0.01,alpha=0.001,iters=2000,variance=0.001,samples=32):

    model.eval()
    final_adv = []
    with torch.no_grad():
        BATCH_SIZE = data.shape[0]
        for b in range(BATCH_SIZE):
            adv_data=torch.squeeze(data[b].clone())
            labels = torch.ones_like(labels_og) * labels_og[b]
            adv_data.detach()
            adv_data_og =  adv_data.clone()
            for _ in range(iters):
                est_g = torch.zeros_like(adv_data)
                for j in range(samples // BATCH_SIZE):
                    adv_data_repeat = adv_data.repeat([BATCH_SIZE,1,1])
                    pert = torch.normal(0.0,1.0,size=adv_data_repeat.shape).cuda()
                    
                    adv_data_repeat_1 = adv_data_repeat + pert * variance
                    logits_1,_,_ = model(adv_data_repeat_1)
                    loss_1 = cal_loss(logits_1,None,labels)

                    adv_data_repeat_2 = adv_data_repeat - pert * variance
                    logits_2,_,_ = model(adv_data_repeat_2)
                    loss_2 = cal_loss(logits_2,None,labels)

                    est_g += torch.sum((loss_1 - loss_2) * pert / (2 * variance),0)
                
                est_g = est_g / samples
                adv_data = adv_data + alpha * est_g.sign()
                delta = adv_data-adv_data_og
                delta = torch.clamp(delta,-eps,eps)
                adv_data = adv_data_og+delta
            final_adv.append(adv_data)

        final_adv = torch.stack(final_adv)
    
    return final_adv.cuda()