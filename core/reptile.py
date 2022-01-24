import datetime, os
now = datetime.datetime.now
import numpy as np
import matplotlib.pyplot as plt
import torch
from .pies import  dqn_pie, zero_pie
from .train import train_live_policy


def reptile_train(policy, rolicy, path, params, 
                    common_infra, explorerL, P=print, F=lambda x,y: None):
    diff_hist=[]
    meta_lr_hist = []
    stamp=now()
    tempsave = 'reptile.pie'
    cp_freq = params.cp_freq
    meta_lr = params.meta_lr
    meta_lr_decay_freq = params.meta_lr_decay_freq
    meta_min_lr = params.meta_min_lr
    meta_lr_decay_ratio =    (params.meta_min_lr/params.meta_lr)**( 1/(params.outer_epochs/params.meta_lr_decay_freq)) 
    inner_diff = dqn_pie(0)
    for outer_epoch in range(params.outer_epochs):
        P('OE:',outer_epoch+1,'of', params.outer_epochs )
        policy.save_external(tempsave)
        
        if (outer_epoch+1)%params.meta_lr_decay_freq==0:
            meta_lr*=meta_lr_decay_ratio
        #param_hist.append((eps, lr))
        meta_lr_hist.append(meta_lr)

        zero_pie(inner_diff)
        for inner_epoch in range(params.inner_epochs): # sample mini batch of tasks
    
            P('\tIE:',inner_epoch+1,'of', params.inner_epochs )
            common_infra.randomize( rtype=(inner_epoch%9) )  #<---- new task (random environment)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
            inner_pie = dqn_pie(0)
            inner_pie.load_external(tempsave)
            
            live_loss_h, live_val_h, live_param_h = \
                train_live_policy(
                    explorerL=explorerL, 
                    policy=inner_pie, 
                    rolicy=rolicy, 
                    path="", 
                    validationS=[], 
                    paramS=params, 
                    reset_exp=True, 
                    P=P, F=lambda x,y: None, S=lambda n1, n2:None)

            # capture the diffrence
            with torch.no_grad():
                difQ, inQ, polQ = inner_diff.Q.parameters(), inner_pie.Q.parameters(), policy.Q.parameters()
                for i,(diff_p, inner_p, policy_p) in enumerate(zip(difQ, inQ, polQ)):
                    diff_p  += (policy_p - inner_p)

        with torch.no_grad():
            difQ, polQ = inner_diff.Q.parameters(), policy.Q.parameters()
            for i,(diff_p, policy_p) in enumerate(zip(difQ, polQ)):
                policy_p  -= torch.mul(meta_lr, diff_p)
               
        if ((outer_epoch+1)%cp_freq)==0:

            if path!="":
                policy.save_external(path)
                P('Checkpoint Created:', now())
            
            diff_mag = torch.tensor([0], dtype=torch.float32)
            with torch.no_grad():
                difQ = inner_diff.Q.parameters()
                for diff_p in difQ:
                    diff_mag += torch.sum(diff_p**2)
            now_diff = diff_mag.item()
            P('Difference:', diff_mag.shape, now_diff)
            diff_hist.append(now_diff)

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









