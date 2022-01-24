import datetime, os
now = datetime.datetime.now
import numpy as np
import matplotlib.pyplot as plt
import torch
from .pies import  dqn_pie, zero_pie
from .train import train_live_policy


def bird_train(policy, rolicy, path, params, 
                    reset_meta_exp, explorerL, P=print, F=lambda x,y: None):
    diff_hist=[]
    meta_lr_hist = []
    stamp=now()
    tempsave = 'bird.pie'
    cp_freq = params.cp_freq
    meta_lr = params.meta_lr
    #meta_lr_decay_freq = params.meta_lr_decay_freq
    #meta_min_lr = params.meta_min_lr
    meta_lr_decay_ratio =    (params.meta_min_lr/params.meta_lr)**( 1/(params.outer_epochs/params.meta_lr_decay_freq)) 
    #inner_diff = dqn_pie(0)
    #diff_hist=[]
    for outer_epoch in range(params.outer_epochs):
        P('OE:',outer_epoch+1,'of', params.outer_epochs )
        policy.save_external(tempsave)
        
        if (outer_epoch+1)%params.meta_lr_decay_freq==0:
            meta_lr*=meta_lr_decay_ratio
        #param_hist.append((eps, lr))
        meta_lr_hist.append(meta_lr)

        #zero_pie(inner_diff)
        inner_pies = []
        params.inner_epochs = len(explorerL)
        for inner_epoch in range(params.inner_epochs ): # sample mini batch of tasks
    
            P('\tIE:',inner_epoch+1,'of', params.inner_epochs )
            #common_infra.randomize( rtype=(inner_epoch%9) )  #<---- new task (random environment)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
            inner_pie = dqn_pie(0)
            inner_pie.load_external(tempsave)
            
            live_loss_h, live_val_h, live_param_h = \
                train_live_policy(
                    explorer=explorerL[inner_epoch], 
                    policy=inner_pie, 
                    rolicy=rolicy, 
                    path="", 
                    validationS=[], 
                    paramS=params, 
                    reset_exp=reset_meta_exp, 
                    P=P, F=lambda x,y: None, S=lambda n1, n2:None)

            inner_pies.append(inner_pie)

            # capture the diffrence
            #with torch.no_grad():
            #    difQ, inQ, polQ = inner_diff.Q.parameters(), inner_pie.Q.parameters(), policy.Q.parameters()
            #    for i,(diff_p, inner_p, policy_p) in enumerate(zip(difQ, inQ, polQ)):
            #        diff_p  += (policy_p - inner_p)

        
        
        P('# 10: meta update')
        # find the difference vector
        for meta_epoch in range(params.meta_epochs):
            #policy.zero_grad()
            diff_stack = []
            for Ti in inner_pies:
                diff_i = []
                for ti,th in zip(Ti.Q.parameters(), policy.Q.parameters()):
                    diff_i.append(torch.mean((ti-th)**2))
                #diff_i_tensor = torch.stack(diff_i)
                diff_stack.append(torch.sum(torch.stack(diff_i)))
            diff_tensors = torch.sum(torch.stack(diff_stack))
            
            diff_hist.append(diff_tensors.item())
            #P('\t Diff:', diff_hist[-1])

            ograds = torch.autograd.grad(diff_tensors, policy.Q.parameters(), create_graph=False)
            #P('... update outer params')
            with torch.no_grad():
                #grad_sum = torch.sum(ograds)
                #P('Outer-Grads:', grad_sum.item())
                for t_param, grad in zip(policy.Q.parameters(), ograds):
                    #print( 'grad-shapes', t_param , grad )
                    t_param -= params.meta_lr * grad
                    #print( 'After', t_param )






        P('\t Diff:', diff_hist[-1])   
        if ((outer_epoch+1)%cp_freq)==0:

            if path!="":
                policy.save_external(path)
                P('Checkpoint Created:', now())
                
            

    fig = plt.figure(figsize=(16,4))
    #S(npy, npy_name)
    plt.plot(diff_hist, color='red', linewidth=0.7)
    plt.ylabel('Difference History')
    plt.grid(axis='both')
    plt.show()
    F(fig,'meta_loss')
    plt.close()
    
    fig = plt.figure(figsize=(16,4))
    #S(npy, npy_name)
    plt.plot(meta_lr_hist, color='green', linewidth=0.7)
    plt.ylabel('LR History')
    plt.grid(axis='both')
    plt.show()
    F(fig,'meta_lr')
    plt.close()
    
    
    P('Finished! Elapsed Time:', now()-stamp )
    if path!="":
        policy.save_external(path)
        P('Saved Meta-Policy @', path)
    
    return









