import numpy as np
import json

class PIE:
    """ Implements Dictionary-Based Q-Learning 
    
        Q(s,a) = (1-lr)*Q(s,a) + lr*(R(s,a,s') + dis*maxQ(s',A)) 
        
    
        Initialize new Q-Learner on discrete action space 
        
        Args:
            lr              learning rate  
            ls              learn steps (usually 1)
            dis             discount factor 
            nos_actions     no of discrete actions (0 to acts-1)
            mapper          ptional { function: state(array) --> state(string) }
                            .. mapper is a function that returns the string representation 
                            .. of a state vector to be stored in dict keys 

        # target = reward + self.dis * max(self.Q[nS][0]) * int(not done) 
        # Note: we should store Q-Values as 
        #  { 'si' : [ Q(si,a1), Q(si,a2), Q(si,a3), ... ] }
        # but we also want to store the number of times a state was visited
        # instaed of creating seperate dictionary, we use same dict and store #visited in position
        #  { 'si' : [ [ Q(si,a1), Q(si,a2), Q(si,a3), ... ], #visited ] }
        """
        
    def __init__(self, nos_actions, lr=0.5, dis=1, mapper=str, seed=None):
        self.lr, self.dis = lr, dis           
        self.A = nos_actions
        self.Q={}                 # the Q-dictionary where Q-values are stored
        self.mapper = mapper      # a function that mapps state vectors to strings
        self.train_count = 0      # counts the number of updates made to Q-values
        self.random = np.random.default_rng(seed)
        
    def predict(self, state):
        cS = self.mapper(state)
        if not cS in self.Q:
            action = self.random.integers(0, self.A, size=1)[0]
        else:
            qvals = (self.Q[cS][0])
            qis = np.where(qvals==np.max(qvals))[0]
            action = self.random.choice(qis, size=1)[0]
        return action
        
    def QVT(self, states):
        qList=[]
        for state in states:
            cS = self.mapper(state)
            if not cS in self.Q:
                print('WARNING:TQL-LEARN:: not found',cS )
                qvals = [0 for _ in range(self.A)] #  [0 for _ in range(self.A)]  ,  np.zeros(self.A)
            else:
                qvals = self.Q[cS][0]
            qList.append(qvals)
        return qList
        
    def learn(self, memory, batch):
        """ memory = explorer's memory
            batch = indices in memomry, use following
                    batch = memory.recent(batch_size)
                    batch = memory.sample(batch_size)
                    batch = memory.all()
        """
        # batch = memory.recent(batch_size) # sample most recent 
        for i in batch:
            cSx, nSx, act, reward, done, _ = memory.mem[i] 
            cS, nS = self.mapper(cSx), self.mapper(nSx)
            #print('cS', type(cS), cS)
            #print('Q', type(self.Q), self.Q)
            #print('A', type(self.A), self.A)
            if not cS in self.Q:
                #print('LEARN:: not found',cS )
                self.Q[cS] = [[0 for _ in range(self.A)], 1, cSx]
            else:
                self.Q[cS][1]+= 1
            if not nS in self.Q: 
                self.Q[nS] = [[0 for _ in range(self.A)], 0, nSx]
                
            self.Q[cS][0][act] = (1-self.lr)    *   ( self.Q[cS][0][act] ) + \
                                 (self.lr)      *   ( reward + self.dis * np.max(self.Q[nS][0]) * int(not done) )  #<--- Q-update
            self.train_count+=1
        return 

    def clear(self):
        self.Q.clear()
        self.train_count=0
        return

    def render(self, mode=0, P=print):
        """ use mode=1 to view full dictionary """
        res='=-=-=-=-==-=-=-=-=\nDICT: Q-Values  #'+str(len(self.Q))+'\n=-=-=-=-==-=-=-=-=\n'
        if mode>0:
            for x,i in enumerate(self.Q):
                if mode>1:
                    rep = str(self.Q[i][2])
                else:
                    rep = str(x)
                res+= rep + '\t\t' + str(self.Q[i][0:2]) + '\n'
            res = res + '=-=-=-=-==-=-=-=-='
        P(res)
        return 
        
    def save(self, path):
        f=open(path, 'w')
        f.write(json.dumps(self.Q, sort_keys=False, indent=4))
        f.close()
        return
        
    def load(self, path):
        f=open(path, 'r')
        self.Q = json.loads(f.read())
        f.close()
        return

#-------------------