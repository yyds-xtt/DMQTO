import datetime
now = datetime.datetime.now
#import numpy as np
#import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim


import relearn.pies.dqn as DQN
#import relearn.pies.dqn2 as DQN2
#import relearn.pies.tql as TQL
import relearn.pies.rnd as RND
from .params import G_ENV_SHAPE, G_LAYERS, G_ENV_ACTION

#--------------------#--------------------
# Policy Definition
#--------------------#--------------------

def dqn_pieX(lrx, state_shape, layer_list, action_shape):
        return DQN.PIE(
                state_dim=state_shape, 
                LL=layer_list, 
                action_dim=action_shape, 
                opt=optim.Adam,
                cost=nn.MSELoss, 
                lr=lrx, 
                mapper=lambda x:x, 
                double=False, 
                tuf=0,  
                device='cpu', 
                )

def rand_pieX (action_shape, rnd_seed): 
    return RND.PIE(action_shape, seed=rnd_seed)

def zero_pie(p):
    pie_params = p.Q.parameters()
    with torch.no_grad():
        for i,pp in enumerate(pie_params):
            pp.data*=0
        p.Q.eval()
    return

dqn_pie = lambda lrx: dqn_pieX(lrx, G_ENV_SHAPE, G_LAYERS, G_ENV_ACTION)
rand_pie = lambda rseed: rand_pieX (G_ENV_ACTION, rseed)






