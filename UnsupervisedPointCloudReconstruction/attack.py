'''
Description: 
Autor: Jiachen Sun
Date: 2021-04-07 01:26:14
LastEditors: Jiachen Sun
LastEditTime: 2021-04-07 15:05:58
'''
import torch

def pgd_attack(model,data,labels,eps=0.01,alpha=0.0002,iters=50,repeat=1):
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
            outputs,feature_adv = model(adv_data)
            loss = torch.mean(torch.abs(labels-feature_adv))
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

        outputs,feature_adv = model(best_examples)
        loss = torch.mean(torch.abs(labels-feature_adv))
        if loss > max_loss:
            max_loss=loss
            best_examples=adv_data.cpu()
            
    return best_examples.cuda()