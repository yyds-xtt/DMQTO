import numpy as np

class PIE:
    """ Implements Random Policy """

    def __init__(self, nos_actions, seed=None):
        self.A = nos_actions
        self.Q = None
        self.train_count=0
        self.rng = np.random.default_rng(seed)
        
    def predict(self, state):
        return self.rng.integers(0, self.A)

        
    def qvals(self, state):
        qvals = [0 for _ in range(self.A)]
        return qvals
        
    def learn(self, memory, batch):
        self.train_count+=1
        return
    
    def clear(self):
        self.Q = None
        self.train_count=0
        return
        
    def render(self, mode=0, P=print):
        P( "=-=-=-=-==-=-=-=-=\n RANDOM POLICY \n=-=-=-=-==-=-=-=-=" )
    
    def save(self, path):
        return
        
    def load(self, path):
        return
        
#-------------------