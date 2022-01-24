
import os
from typing import OrderedDict
import numpy as np

import torch as T
import torch.nn as nn
#import torch.optim as optim
#import torch.nn.functional as F
def shape2size(shape):
    res = 1
    for d in shape:
        res*=d
    return res  

class Qnetn(nn.Module):
    def __init__(self, state_dim, LL, action_dim):
        super(Qnetn, self).__init__()
        self.n_layers = len(LL)
        if self.n_layers<1:
            raise ValueError('need at least 1 layers')
        layers = [nn.Linear(state_dim, LL[0]), nn.ReLU()]
        for i in range(self.n_layers-1):
            layers.append(nn.Linear(LL[i], LL[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(LL[-1], action_dim))
        self.SEQL = nn.Sequential( *layers )

    def forward(self, x):
        logits = self.SEQL(x)
        return logits

    def info(self, cap="", P=print):
        P('--------------------------')
        P(cap, 'No. Layers:\t', self.n_layers)
        P('--------------------------')
        # print(2*(pie.Q.n_layers+1)) <-- this many params
        std = self.state_dict()
        total_params = 0
        for param in std:
            nos_params = shape2size(std[param].shape)
            P(param, '\t', nos_params, '\t', std[param].shape )
            total_params+=nos_params
        P('--------------------------')
        P('PARAMS:\t', f'{total_params:,}') # 
        P('--------------------------')
        return total_params


class PIE:

    """ 
        Implements DQN based Policy
        
        state_dim       Observation Space Shape
        LL              List of layer sizes for eg. LL=[32,16,8]
        action_dim      Action Space (should be discrete)
        opt             torch.optim     (eg - torch.optim.Adam)
        cost            torch.nn.<loss> (eg - torch.nn.MSELoss)
        lr              Learning Rate for DQN Optimizer ()
        dis             discount factor
        mapper          a mapper from state space to DQN input eg - lambda x: x.reshape(a,b)
        tuf             target update frequency (if tuf==0 then doesnt use target network)
        double          uses double DQN algorithm (with target network)
        device          can be 'cuda' or 'cpu' 
                        #self.device = 'cuda' if T.cuda.is_available() else 'cpu'
        
        Note:
            # single DQN can either be trained with or without target
            # if self.tuf > 0 then it means target T exists and need to updated, otherwise T is same as Q
            # Note that self.T = self.Q if self.tuf<=0
            
        
    """
    
    def __init__(self, state_dim, LL, action_dim, opt, cost, lr, mapper, double=False, tuf=0,  device='cpu'): 
        
        if double and tuf<=0:
            raise ValueError("double DQN requires a target network, set self.tuf>0")
        self.lr = lr
        self.state_dim=state_dim
        self.LL = LL
        self.action_dim=action_dim
        #self.rand = np.random.default_rng(seed)
        self.tuf = tuf
        self.double=double
        self.mapper=mapper
        self.device = device
        self.opt=opt
        self.cost=cost
        
        self.base_model = Qnetn(state_dim, LL, action_dim).to(self.device)
        self.Q = Qnetn(state_dim, LL, action_dim).to(self.device)
        self.T = Qnetn(state_dim, LL, action_dim).to(self.device) if (self.tuf>0) else self.Q
        self.clear()

    def clear(self):
        self._clearQ()
        self.optimizer = self.opt(self.Q.parameters(), lr=self.lr) # opt = optim.Adam
        self.loss_fn = self.cost()  # cost=nn.MSELoss()
        self.train_count=0
        self.update_count=0
    def _clearQ(self):
        with T.no_grad():
            self.Q.load_state_dict(self.base_model.state_dict())
            self.Q.eval()
            if (self.tuf>0):
                self.T.load_state_dict(self.base_model.state_dict())
                self.T.eval()
    def _loadQ(self, from_dqn):
        with T.no_grad():
            self.Q.load_state_dict(from_dqn.Q.state_dict())
            self.Q.eval()


        
    def predict(self, state):
        st = T.tensor(self.mapper(state), dtype=T.float32)
        qvals = self.Q(st)
        m,i =  T.max(  qvals , dim=0  )
        return i.item()

    def _prepare_batch(self, memory, batch):
        #batch = memory.sample(size)
        steps = len(batch)
        cS, nS, act, reward, done = [], [], [], [], []
        nSnp = []
        for i in batch:
            cSi, nSi, acti, rewardi, donei, _ = memory.mem[i]
            cS.append(self.mapper(cSi))
            nS.append(self.mapper(nSi))
            #nSnp.append(self.mapper(nSi))
            act.append(acti)
            reward.append(rewardi)
            done.append(int(donei))
        return  steps, np.arange(steps), \
                T.tensor(cS, dtype=T.float32).to(self.device), \
                T.tensor(nS, dtype=T.float32).to(self.device), \
                np.array(act), \
                T.tensor(reward, dtype=T.float32).to(self.device), \
                T.tensor(done, dtype=T.float32).to(self.device)
                #nSnp


    def fit(self, Qd, max_epochs, lrF, min_loss=0.0, verbose=0, P=print):
        # fit the Q-value dictonary as obtained by tql
        dx, dy = [], []
        for key in Qd:
            val = Qd[key] # [ {Q}, nos_visited, state_vector)
            dx.append(val[2])
            dy.append(val[0])
        # Compute prediction and loss
        dx = T.tensor(dx, dtype=T.float32).to(self.device)
        dy = T.tensor(dy, dtype=T.float32).to(self.device)
        loss_hist, lr_hist = [], []
        for epoch in range (max_epochs):
        
            new_lr = lrF (epoch)
            with T.no_grad():
                for g in self.optimizer.param_groups:
                    g['lr'] = new_lr
            lr_hist.append(new_lr)
            
            
            pred = self.Q(dx)
            target = dy
            loss =  self.loss_fn(pred, target)  #T.tensor()
            self.optimizer.zero_grad()
            loss.backward()
            current_loss = loss.item()
            self.optimizer.step()
            
            loss_hist.append(current_loss)
            if current_loss<=min_loss:
                P('Epoch:',epoch+1,'Reached minimum loss:', current_loss, '/', min_loss)
                break
                

        return loss_hist, lr_hist
        
    def learn(self, memory, batch, lr=0.0, dis=1):
        if lr>0.0:
            self.optimizer.param_groups[0]['lr']=lr
        steps, indices, cS, nS, act, reward, done  = self._prepare_batch(memory, batch)
        
        target_val = self.T(nS) #if type(target_pie)==type(None) else T.tensor(target_pie.QVT(nSnp), dtype=T.float32)
        #print('target_val', target_val.shape, target_val.dtype)
        if not self.double:
            updater, _ = T.max(target_val, dim=1)
        else:            
            _, target_next = T.max(self.Q(nS), dim=1) # tensor.max returns indices as well
            updater=T.zeros(steps,dtype=T.float32)
            updater[indices] = target_val[indices, target_next[indices]]
        updated_q_values = reward + dis * updater * (1 - done)
        
        # Compute prediction and loss
        pred = self.Q(cS)
        target = pred.detach().clone()
        target[indices, act[indices]] = updated_q_values[indices]
        loss =  self.loss_fn(pred, target)  #T.tensor()
        self.optimizer.zero_grad()

        #if reg_l2_lambda>0: # adding L2 regularization
        #    l2_lambda = reg_l2_lambda
        #    l2_norm = sum(p.pow(2.0).sum() for p in self.Q.parameters())
        #    loss = loss + l2_lambda * l2_norm
                        
        # this does not happen
        #target[indices, act[indices]] = updated_q_values[indices]*(self.theta) + target[indices, act[indices]]*(1-self.theta)
        #if lr>0.0:
        #    with T.no_grad():
        #        for g in self.optimizer.param_groups:
         #           g['lr'] = lr               
        # Backpropagation

        #for param in self.Q.parameters():
        #    param.grad.data.clamp_(-1, 1)  # clip norm <-- dont do it
        
        #if do_step:
        #grads=None
        loss.backward()
        self.optimizer.step()
        self.train_count+=1
        #else:
           # grads = T.autograd.grad(loss, self.Q.parameters(), create_graph=True)
            #grads=[ param.grad.data.detach().clone() for param in self.Q.parameters() ]
            #self.optimizer.zero_grad()

        if (self.tuf>0):
            if self.train_count % self.tuf == 0:
                self.update_target()
                self.update_count+=1
        return loss.item() #grads

    def update_target(self):
        with T.no_grad():
            self.T.load_state_dict(self.Q.state_dict())
            self.T.eval()

    def render(self, mode=0, P=print):
        P('=-=-=-=-==-=-=-=-=\nQ-NET\n=-=-=-=-==-=-=-=-=')
        P(self.Q)
        if mode>0:
            self.Q.info(cap='[LAYER INFO]',P=P)
        #P('\nPARAMETERS\n')
        #Q = self.Q.parameters()
        #for i,q in enumerate(Q):
        #    print(i, q.shape)
        P('Train Count:', self.train_count)
        if (self.tuf>0):
            P('Update Count:', self.update_count)
        P('=-=-=-=-==-=-=-=-=!Q-net=-=-=-=-==-=-=-=-=')
        return
        
    #def save(self, filename):
    #    T.save(self.Q, os.path.join(self.path,filename))
        
    #def load(self, filename):
    #    self.base_model = T.load(os.path.join(self.path,filename))
    #    self._clearQ()
    #    return
        
    def save_external(self, filename):
        T.save(self.Q, filename)

    def load_external(self, filename):
        self.base_model = T.load(filename)
        self._clearQ()
        return