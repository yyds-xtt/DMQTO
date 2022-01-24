# %% [markdown]
# # global

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
from core.bird import *

# %% [markdown]
# # args

# %%

args=None
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #now_ts = basic.stamp_now()
    parser.add_argument('--alias', type=str, default='META', help='')
    parser.add_argument('--seed', type=int, default=-1, help='')


    parser.add_argument('--meta_lr', type=float, default=0.1, help='')
    parser.add_argument('--meta_lr_decay_freq', type=int, default=20 , help='')
    parser.add_argument('--meta_min_lr', type=float, default=0.005 , help='')
    parser.add_argument('--meta_epochs', type=int, default=10000, help='')
    parser.add_argument('--outer_epochs', type=int, default=5, help='')
    #parser.add_argument('--inner_epochs', type=int, default=9, help='')
    parser.add_argument('--cp_freq', type=int, default=1, help='')


    parser.add_argument('--epochs', type=int, default=10_000, help='')
    parser.add_argument('--min_eps', type=float, default=0.2, help='')
    parser.add_argument('--episodic_K', type=int, default=1, help='')
    parser.add_argument('--explore_K', type=int, default=64, help='')
    parser.add_argument('--min_mem_K', type=int, default=640, help='')
    parser.add_argument('--batch_K', type=int, default=32, help='')
    parser.add_argument('--steps_K', type=int, default=1, help='')
    parser.add_argument('--initial_lr', type=float, default=0.0100 , help='')
    parser.add_argument('--min_lr', type=float, default=0.0001 , help='')
    parser.add_argument('--lr_decay_freq', type=int, default=20 , help='')
    parser.add_argument('--test_freq', type=int, default=10_000, help='')

    #parser.add_argument('--exp_nos', type=int, default=2, help='')
    parser.add_argument('--mem_cap', type=int, default=100_000, help='')
    args = parser.parse_args()


# %% [markdown]
# # common

# %%

#------------------------------------------
# Common Params
#------------------------------------------
XP_ALIAS =              args.alias #HINT stamp_now()
GLOBAL_RNG_SEED =       np.random.randint(1,100000) if args.seed<0 else args.seed #HINT #<-[rng]
#------------------------------------------
#NOS_EXPLOERS =      args.exp_nos 
EXP_CAP =           args.mem_cap #HINT: MAX Training

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
# 
# # Training - common infra, different flowgens
# 
# 

# %%

#--------------------#--------------------
# Training - common infra, different flowgens
#--------------------#--------------------
exp_infra =     dbinfra()
exp_infra.render(P=P)

meta_profiles = ( 
    APP_PROFILES['app_1']['pro_1'],
    APP_PROFILES['app_2']['pro_1'],
    APP_PROFILES['app_3']['pro_1'],
    APP_PROFILES['app_4']['pro_1'],
    APP_PROFILES['app_5']['pro_1'],
    APP_PROFILES['app_1']['pro_2'],
    APP_PROFILES['app_2']['pro_2'],
    APP_PROFILES['app_3']['pro_2'],
    APP_PROFILES['app_4']['pro_2'],
    APP_PROFILES['app_5']['pro_2'],
 )
expL = [ EXP (get_env(exp_infra, FLOWGEN(ap_task, ap_levels, seed=randint())), 
                    cap=EXP_CAP, mseed=randint(), seed=randint()) \
                    for ap_task, ap_levels in meta_profiles ]


# %% [markdown]
# # policy

# %%

#-====================================
# Policy
#-====================================
meta_policy = dqn_pie(0)
P(basic.strD(meta_policy.__dict__, sep='\n'))
meta_policy.render(mode=1, P=P)
#-====================================

# %% [markdown]
# # meta

# %%

#-====================================
# Meta-Training
#-====================================

bird_train(
    policy=meta_policy, 
    rolicy=rand_pie(randint()), 
    path=os.path.join(XP_ALIAS,'meta'), 
    params=args,
    reset_meta_exp=True, 
    explorerL=expL, 
    P=print, 
    F=lambda x,y: None)


