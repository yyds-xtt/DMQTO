
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
#---------------------------------------------------------
# Defines EXP and MEM
#---------------------------------------------------------


class EXP:
    """ An explorer that interacts with the environment
    """
    
    def __init__(self, env, cap, mseed=None, seed=None):
        self.env = env
        self.A = self.env.action_space.n
        self.cS, self.done = self.env.reset(), False  #<---------- calls reset once at initialization
        self.memory = MEM(capacity=cap, seed=mseed)
        self.random = np.random.default_rng(seed)
    
    def reset(self, clear_mem=False):
        self.cS, self.done = self.env.reset(), False
        if clear_mem:
            self.memory.clear()
        return
            
    def step(self, greedy_pie, rand_pie, epsilon, test=False):
        """ explore for one step """
        
        # test v/s explore
        if test:
        
            # 1. Choose action          #<--- always use policy to predict actions
            act = greedy_pie.predict(self.cS)
            
            # 2. Step in enviroment
            nS, reward, self.done, _ = self.env.step(act)   

            # 3. Prepare Transition #<<--- as per MEM.DEFAULT_SCHEMA (do not pass state vectors)
            transition = (None, None, act, reward, self.done, self.env.get_tag()) 
            
        else:
            
            # 1. Choose action          #<--- use random actions or policy to predict actions based on epsilon
            # NOTE: use self.random instead of self.env.action_space.sample()
            # self.random.integers(0, self.A) \
            act =  rand_pie.predict(self.cS) \
                if (self.random.random(size=1)[0] < epsilon) \
                else greedy_pie.predict(self.cS)
                
            # 2. Step in enviroment
            cS = self.cS 
            nS, reward, self.done, _ = self.env.step(act)
            
            # 3. Prepare Transition #<<--- as per MEM.DEFAULT_SCHEMA (pass all state vectors)
            transition = (cS, nS, act, reward, self.done, self.env.get_tag())     
        
        # common part
        
        # 4. Store transition to memory
        self.memory.commit(transition)
        
        # 5. Reset in final state
        if self.done or self.env._elapsed_steps>=self.env._max_episode_steps: 
            self.reset()
            self.memory.mark() #<--- mark episode
            done=True
        else:
            self.cS = nS
            done=False
        return done  #<----- this 'done' is different that 'self.done' as it is true when _max_episode_steps has elapsed and env might not be in final state
    
    def episode(self, greedy_pie, rand_pie, epsilon, test=False):
        """ explore for one episode """
        done, ts = False, 0
        while not done:
            done = self.step(greedy_pie, rand_pie, epsilon, test=test)
            ts+=1
        return ts
    
    def explore(self, 
                greedy_pie, 
                rand_pie,
                moves, 
                epsilonF,
                episodic=False, 
                test=False):
        """ explore for given moves with decaying epsilon 
            
        Args:
            pie         Policy for exploration (greedy action)
            moves       explore for given moves - which can be either 'steps' or 'episodes'
            episodic    if True, moves means 'episodes' else it means 'steps'
            test        if True, always use policy(pie) to take action and do not store state vectors in memory

        """
        if episodic:
            ts = 0
            for k in range(moves):
                t = self.episode(greedy_pie, rand_pie, epsilonF(moves), test=test) # t is integer
                ts += t
        else:
            ts = moves
            for k in range(moves):
                t = self.step(greedy_pie, rand_pie, epsilonF(moves), test=test) # t is boolean
        return ts
        
       
        
    def summary(self, P=print):
        """ prepares summary based on memory - useful in testing 
            set P=lambda *x: None to not print """
        
        # choose all transactions from memory and read cols 2 to 5 i.e ('Action', 'Reward',  'Done', 'tag')
        npe = np.array(self.memory.read_cols(self.memory.all(), 2, 6 ))
        clean_up=False
        # assume that memory.mark is corrrectly called, find the episode markers
        if len(self.memory.episodes)==0:
            self.memory.episodes.append(self.memory.count)
            clean_up=True
        else:
            if self.memory.episodes[-1]!=self.memory.count:
                self.memory.episodes.append(self.memory.count)
                clean_up=True
        
        si = 0
        cnt = 0
        header = np.array(['Episode', 'Start', 'End', 'Steps', 'Reward', 'Done', 'Tag'])
        rows= []
        aseqs = []
        for ei in self.memory.episodes:
            cnt+=1
            ep = npe[si:ei] # ep = [action, reward, done]
            aseq, rsum, etag = ep[:,0], np.sum(ep[:,1]), ep[-1,3] # action sequence, total reward
            row = [cnt, si, ei, len(aseq), rsum, int(ep[-1,2]), etag]
            rows.append(row)
            aseqs.append(aseq)
            si = ei
        if clean_up:
            del self.memory.episodes[-1]
        rows = np.array(rows)
        aseqs = np.array(aseqs)
        avg_reward =  np.mean(rows[:, 4])
        
        #  print results
        P('==========================================================\n')
        hd=""
        for i in header:
            hd+=(i+ '\t')
        P(hd)
        for i in range(len(rows)):
            rowstr = ""
            for j in range(len(rows[i])):
                rowstr += str(rows[i][j]) + '\t' 
            P(rowstr ,':\t:', aseqs[i])
        P('\n==========================================================')
        return header, rows, avg_reward, aseqs
#---------------------------------------------------------

#---------------------------------------------------------

class MEM:
    """ A list based memory for explorer """
    
    DEFAULT_SCHEMA = ('cS', 'nS', 'Action', 'Reward',  'Done', 'Tag')
    
    def __init__(self, capacity, seed=None):
        self.capacity = capacity
        self.mem = []
        self.episodes=[]
        self.count = 0
        self.random = np.random.default_rng(seed)
               
    def clone(self):
        res = MEM(self.capacity)
        res.mem = deepcopy(self.mem)
        res.count = deepcopy(self.count)
        res.episodes = deepcopy(self.episodes)
        return res
        
    def set_seed(self, seed):
        self.random = np.random.default_rng(seed)
    
    def clear(self):
        self.mem.clear()
        self.episodes.clear()
        self.count=0
        return
        
    def commit(self, transition): 
        if self.count>=self.capacity:
           del self.mem[0:1]
        else:
            self.count+=1
        self.mem.append(transition)
        return

    def mark(self):
        self.episodes.append(self.count)
        return
        
    def sample(self, batch_size):
        batch_pick_size = min(batch_size, self.count)
        return self.random.integers(0, self.count, size=batch_pick_size)
        
    def recent(self, batch_size):
        batch_pick_size = min(batch_size, self.count)
        return np.arange(self.count-batch_pick_size, self.count, 1)
        
    def all(self):
        return np.arange(0, self.count, 1)
        
    def read(self, iSamp):
        return [ self.mem[i] for i in iSamp ]
        
    def read_col(self, iSamp, iCol):
        return [ self.mem[i][iCol] for i in iSamp ]
        
    def read_cols(self, iSamp, iCol_from, iCol_to):
        return [ self.mem[i][iCol_from:iCol_to] for i in iSamp ]

    def render(self, low, high, step=1, p=print):
        p('=-=-=-=-==-=-=-=-=@MEMORY=-=-=-=-==-=-=-=-=')
        p("Status ["+str(self.count)+" | "+str(self.capacity)+ "]")
        p('------------------@SLOTS------------------')
        
        for i in range (low, high, step):
            p('SLOT: [', i, ']')
            for j in range(len(self.mem[i])):
                p('\t',MEM.DEFAULT_SCHEMA[j],':', self.mem[i][j])
            p('-------------------')
        p('=-=-=-=-==-=-=-=-=!MEMORY=-=-=-=-==-=-=-=-=')
        return     
        
    def render_all(self, p=print):
        self.render(0, self.count, p=p)
        return
        
    def render_last(self, nos, p=print):
        self.render(-1, -nos, step=-1,  p=p)
        return


#---------------------------------------------------------
