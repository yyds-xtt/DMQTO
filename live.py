# %%


# %% [markdown]
# # Global

# %%
import math, sys, datetime, argparse, random, os
now = datetime.datetime.now
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from mdlog.logger import MDLOG

import relearn.pies.dqn as DQN
#import relearn.pies.dqn2 as DQN2
#import relearn.pies.tql as TQL
import relearn.pies.rnd as RND
from relearn.explore import EXP, MEM

import snet.infra as db
import snet.basic as basic
#import snet.utils as utils
from snet.flow import FLOW, FLOWGEN, COST, BASELINE
from snet.fenv import  ENV
from core.params import *
from core.pies import *
from core.train import *

# %% [markdown]
# # args

# %%


args=None
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #now_ts = basic.stamp_now()
    parser.add_argument('--alias', type=str, default='LIVE_RAND', help='')
    parser.add_argument('--seed', type=int, default=-1, help='')

    parser.add_argument('--epochs', type=int, default=8000, help='')
    parser.add_argument('--min_eps', type=float, default=0.2, help='')
    parser.add_argument('--episodic_K', type=int, default=1, help='')#<------ based on
    parser.add_argument('--explore_K', type=int, default=25, help='')#<------ episode/steps
    parser.add_argument('--min_mem_K', type=int, default=200, help='')
    parser.add_argument('--batch_K', type=int, default=25, help='') #<------ steps
    parser.add_argument('--steps_K', type=int, default=1, help='')
    parser.add_argument('--initial_lr', type=float, default=0.0100 , help='')
    parser.add_argument('--min_lr', type=float, default=0.0001 , help='')
    parser.add_argument('--lr_decay_freq', type=int, default=20 , help='')
    parser.add_argument('--test_freq', type=int, default=100, help='')

    #parser.add_argument('--exp_nos', type=int, default=2, help='')
    parser.add_argument('--val_nos', type=int, default=100, help='')
    #parser.add_argument('--rand_times', type=int, default=0, help='')
    parser.add_argument('--app', type=str, default="app_1", help='')
    parser.add_argument('--pro', type=str, default="pro_1", help='')
    parser.add_argument('--mem_cap', type=int, default=50_000, help='')#<------
    parser.add_argument('--load_policy', type=int, default=0, help='')
    parser.add_argument('--load_alias', type=str, default='META', help='')
    args = parser.parse_args()


# %% [markdown]
# # Common Params

# %%
#------------------------------------------
# Common Params
#------------------------------------------
XP_ALIAS =              args.alias  #'LEARN' #HINT stamp_now()
GLOBAL_RNG_SEED =       np.random.randint(1,100000) if args.seed<0 else args.seed #HINT #<-[rng]
#-------------------------------------
#-====================================
#NOS_EXPLOERS =      args.exp_nos 
#META_VALIDATOR =    False            #HINT: True during meta-learning, validates on random environments
val_nos =           args.val_nos              #HINT: val_nos # no of validation workflows
#RAND_TIMES =        args.rand_times              #HINT: (optional) <--- randomize once (not nessesary for meta-learning)
EXP_CAP =            args.mem_cap #HINT: MAX Training
LIVE_POLICY_LOAD =      bool(args.load_policy) # random or meta(load from disk)
LIVE_POLICY_PATH =      os.path.join(args.load_alias,'meta')
#-====================================

# define common functions
P = print
F = lambda fig, file_name: fig.savefig(os.path.join(XP_ALIAS, file_name)) if AUTOSAVE else None
S = lambda npy, file_name: np.save(os.path.join(XP_ALIAS, file_name), npy) if AUTOSAVE else None
_ = os.makedirs(XP_ALIAS, exist_ok=True) if AUTOSAVE else None
prng = np.random.default_rng(seed=GLOBAL_RNG_SEED) # RNG for seed generation
randint = lambda : prng.integers(1,10000)

P(XP_ALIAS,'@' ,now())
P('GLOBAL_RNG_SEED:',GLOBAL_RNG_SEED)
P('State-Space:{}, Action-Space:{}'.format(G_ENV_SHAPE, G_ENV_ACTION))

# %% [markdown]
# # envs

# %%

#--------------------#--------------------
# Training - common infra, different flowgens
#--------------------#--------------------
exp_infra =     dbinfra()
#exp_infra.randomize(RAND_TIMES) 
exp_infra.render(P=P)
ntask, nlevels = APP_PROFILES[args.app][args.pro]
print('App profile:',ntask, nlevels)
exp = EXP(get_env(exp_infra, 
                    FLOWGEN(ntask, nlevels, seed=randint())), 
                    cap=EXP_CAP, mseed=randint(), seed=randint())


#--------------------#--------------------
# Validation
#--------------------#--------------------
FIX_GEN = FLOWGEN(ntask, nlevels, seed=randint()) 
vxpS = []
vinfra = dbinfra()
for v in range(val_nos):
    vxpS.append( EXP(env=get_env(vinfra, FLOWGEN(ntask, None, fixed_flow=FIX_GEN.gen_flow())), 
                     cap=np.inf, mseed=randint(), seed=randint())  )
P('Validation Explorers:', len(vxpS))




# %% [markdown]
# # policy

# %%

#-====================================
# Policy
#-====================================
live_policy = dqn_pie(0)
_=live_policy.load_external(LIVE_POLICY_PATH) if LIVE_POLICY_LOAD else None
P(basic.strD(live_policy.__dict__, sep='\n'))
live_policy.render(mode=1, P=P)
#-====================================



# %% [markdown]
# # train

# %%


#-====================================
# Training
#-====================================
live_loss_h, live_val_h, live_param_h = \
    train_live_policy(
        explorer=exp, 
        policy=live_policy, 
        rolicy=rand_pie(randint()), 
        path=os.path.join(XP_ALIAS,'policy'), 
        validationS=vxpS, 
        paramS=args, 
        reset_exp=True, 
        P=P, F=F, S=S)


