import os
import numpy as np
import matplotlib.pyplot as plt
from .basic import effective_bandwidth, base2int, int2base, strA
#from scipy.sparse.csgraph import floyd_warshall
from .flow import COST
from gym import spaces

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ENV:    base environment
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
class ENV:
    def __init__(self, 
                arg_infra, 
                flow_generator, 
                state_vector_len,
                single_gene = False, 
                name="", 
                initial_reset=False):
        
        self.name = name
        self.single_gene = single_gene      # actions will modify single genes in a solution

        # check and copy flow generators
        self.flowgen = flow_generator              # workflow generators
        self.T = self.flowgen.T
        self.LEN = state_vector_len
        self.dT = self.LEN  - (self.flowgen.LEN + self.T+1) #(self.MAXT - self.T)**2
        self._max_episode_steps=self.T
        self._elapsed_steps = 0

        self.infra = arg_infra
        self.A = arg_infra.A    # - placement locations (devices = 1 + C + E)
        self.action_space = spaces.discrete.Discrete(self.A)
        self.LOC_ = np.zeros(self.T + 1)
        self.MAP = np.arange(0, len(self.LOC_), 1)
        #self.observation_space = spaces.box.Box(low=0, high=np.inf, shape=ospace )
              
        if initial_reset:
            self.reset()

        return

    def get_state(self):
        return np.hstack(( self.LOC_[self.MAP], self.flow.to_aug_npy(self.dT) ))

    def reset(self):
        self.flow = self.flowgen.new_flow()
        return self.restart()

    def restart(self):
        self._elapsed_steps = 0
        self.flow.V[0] = self._elapsed_steps
        self.LOC_[1:] = self.LOC_[0] # <--- same as initial location in the flow

        self._icost=self.current_cost()
        return self.get_state()

    def get_tag(self):
        return self._icost
    def step(self, act):
        assert(self._elapsed_steps<self._max_episode_steps) # <-- assertion error is raised

        prev_cost = self._icost
        t = self._elapsed_steps+1 #self.flow.task_count() #int(self.V_[0])
        if self.single_gene:
            self.LOC_[t] = act
        else:
            self.LOC_[t:] = act
            
        self._icost=self.current_cost()
        self._elapsed_steps+=1
        self.flow.V[0] = self._elapsed_steps
        c = prev_cost-self._icost
        d=(self._elapsed_steps>=self._max_episode_steps)
        return self.get_state(), c, d, None

    def current_cost(self):
        return COST(self.infra, self.flow, self.LOC_.astype(np.int32))

    def render(self, show_infra=False, P=print):
        
        if show_infra:
            P("________________________________________________")
            self.infra.render(P=P)
        P("-------------------------------------------------")
        self.flow.render(P=P)
        #P(self.flows[self.flow_index].render(P=P))
        P("________________________________________________")
        P('|--STEP:\t', self._elapsed_steps)
        P('|--LOC:\t', str(self.LOC_))
        P('|--COST:\t', str(self._icost))
        P("________________________________________________")
        return
