import torch
import numpy as np
np.random.seed(666)
torch.manual_seed(666)
import torch.nn.functional as F
from util import cross_entropy_with_probs, cal_loss

def ATTA_attack(model,data,labels,natural_data,eps=0.01,alpha=0.001,iters=1):
    model.eval()
    adv_data=data.clone()
    adv_data.detach()
    for i in range(iters):
        adv_data.requires_grad=True
        outputs,_,_ = model(adv_data)
        loss = F.cross_entropy(outputs,labels)
        loss.backward()
        with torch.no_grad():    
            adv_data = adv_data + alpha*adv_data.grad.sign()
            delta = adv_data-natural_data
            delta = torch.clamp(delta,-eps,eps)
            adv_data = natural_data+delta
            #If points outside the unit cube are invalid then
            adv_data = torch.clamp(adv_data,-1,1)
    return adv_data.cuda()
 
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
                loss = cal_loss(outputs,trans,labels)
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

        outputs,_,trans = model(adv_data)
        if mixup:
            loss = cross_entropy_with_probs(outputs,labels)
        else:
            loss = cal_loss(outputs,trans,labels)
        if loss > max_loss:
            max_loss=loss
            best_examples=adv_data.cpu()
            
    return best_examples.cuda()
 
