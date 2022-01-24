import os
import numpy as np
import matplotlib.pyplot as plt
from .basic import effective_bandwidth, base2int, int2base, strA
from scipy.sparse.csgraph import floyd_warshall

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# FLOW:    Class for holding Workflow DAG
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
class FLOW:
    
    def __init__(self, flow_npy):
        self.S = flow_npy
        self.L = self.S[0,:]            #<---- L[0] contains total number of levels
        self.V = self.S[1,:]            #<---- V[0] contains total number of tasks
        self.D = self.S[2:,:]

    def task_count(self):
        return len(self.L)-1
        
    def level_count(self):
        return int(self.L[-1])
    def clone(self):
        return FLOW(np.copy(self.S))
    def to_npy(self):
        return self.S.flatten()
    def to_aug_npy(self, delta_aug):

        return np.hstack((self.S.flatten(),np.zeros(delta_aug))) if delta_aug>0 else self.to_npy()
    def geteff(self, r):
        return 1 - (((r)-(self.min_cost))/((self.max_cost)-(self.min_cost)))
    def render(self, P=print):
   
        
        T = self.task_count()
        #L = self.level_count()
        
        ls = ''
        vs = ''
        for x in range(T+1):
            ls+='('+str(int(self.L[x]))+')' + '\t'
            vs+=str(round(self.V[x],2)) + '\t'
        
        P(vs)  
        P(ls)
        
        bs, ds = '', ''
        for x in range(T+1):
            bs+= '['+str(x)+']'+'\t'
            rs = ""
            for y in range(T+1):
                rs+= str(round(self.D[x,y],2)) + '\t'
            ds+=rs+'\t'+'['+str(x)+']\n'
        
        P(bs)
        P(ds)
        return


class FLOWGEN:

    def __init__(self, N_tasks, P_partitions, D_range=(1,100), V_range=(1,60), seed=None, fixed_flow=None):
    
        self.fixed_flow = fixed_flow
        self.isfixed_flow = (type(self.fixed_flow) != type(None))

        self.T =        N_tasks
        self.D_low, self.D_high = D_range
        self.V_low, self.V_high = V_range
        
        if not self.isfixed_flow:
            self.rng = np.random.default_rng(seed) #self.rng.seed
            #----------------------------------------------------
            self.L =        len(P_partitions)
            self.L_nos =    np.hstack((np.array([1]), np.array(P_partitions) )).astype(np.int32)
            assert(np.sum(self.L_nos)==self.T+1)
            self.L_sum =    np.zeros(len(self.L_nos)+1, dtype=np.int32)
            for i in range(len(self.L_sum)):
                self.L_sum[i] = np.sum(self.L_nos[0:i])
            self.L_index = np.zeros(self.T+1)
            t = 1
            for l in range(1, self.L+1):
                nl = self.L_nos[l]
                for _ in range(nl):
                    self.L_index[t] = l
                    t+=1
            #----------------------------------------------------

        # end if
        self.dim =      (self.T+3, self.T+1)
        self.LEN =      (self.T+3)*(self.T+1)
        return



    def new_flow(self):
        if self.isfixed_flow:
            return self.fixed_flow
        else:
            return self.gen_flow()
            
            
    def gen_flow(self):
        flow = FLOW(np.zeros(self.dim))
        flow.L[0:] = self.L_index[0:] #<--- constant
        flow.V[0] = self.T # time step
        flow.V[1:] = self.rng.integers(self.V_low, self.V_high, size=self.T)
        
        # set initial/final node
        flow.D[0, self.L_sum[-2]:] = self.rng.integers(self.D_low, self.D_high, size=(self.L_nos[-1]))
        for level in range(1, self.L+1):
            l_tasks = self.L_nos[level]
            p_tasks = self.L_nos[level-1]
            pl = self.L_sum[level-1]
            cl = self.L_sum[level]
            nl = self.L_sum[level+1] 
            d0 = []
            #if(l_tasks<=0):
                #print('LEVEL:',level)
                #print('L_nos:',self.L_nos)
            for _ in range(l_tasks):
                dzeros = np.zeros(p_tasks) #<--- at least one 1 
                dent_size= self.rng.integers(0, p_tasks) + 1 #if  p_tasks>1 else 1
                dnos = self.rng.choice(np.arange(0,p_tasks,1), size=dent_size, replace=False ) # choose indices
                dzeros[dnos] = 1
                d0.append(dzeros)
            #print(d0)
            d0 = np.array(d0)
            # check coloms of d0 must not be all zero
            for c in range(p_tasks):
                if len(np.where(d0[:, c]==0)[0]) == l_tasks:
                    d0[self.rng.integers(0, l_tasks), c] = 1
            # finally populate self.D
            flow.D[cl:nl, pl:cl]  = np.multiply(self.rng.integers(self.D_low, self.D_high, size=(l_tasks, p_tasks)), d0 )

            
        return flow
    

def COST(infra, workflow, solution, NZ_EPSILON = 0.00001): #<---- make sure solution is int type
    # based on solution (locations), caluate time delay and energy cost
    
    # a temporary data array to build a dag
    temp = np.zeros_like(workflow.D)
    temp[:,:] = np.inf # initial flow as inf
    #DRx, DEx, VRx, VEx = infra
    T = workflow.task_count()
    #print('Task size is ', T)
    max_level = int(workflow.L[-1])+1
    Energy = 0
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for task in range(1, T+1):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        level = int(workflow.L[task])
        iloc = solution[task]
        # P('\nTASK:',task, 'of', T, '@level', level, '@location', iloc)
        
        DR, DE, VR, VE = infra.get_info(iloc)
        #DR, DE, VR, VE = DR[iloc], DE[iloc], VR[iloc], VE[iloc]  #(iloc)
        #P('DR:{}, DE:{}, VR:{}, VE:{}'.format(DR, DE, VR, VE))
        
        # calculate
        
        # # task 
        task_size = workflow.V[task]  #flow[0,1:]
        time_compute = task_size * (VR)
        eng_compute = task_size * (VE)
        
        # # input data
        din = np.where(workflow.L==level-1) [0]
        task_din = workflow.D[task, din] 
        lin = solution[din] #P('LIN:', lin)
        eng_data_in = np.sum(task_din) * (DE)
        time_data_in =  [ task_din[i] * DR[int(lin[i])] for i in range(len(din)) ] 
        # set on temp
        temp[task, din] = time_data_in + time_compute   #P('changing-in', task, din)

        # # output data
        dout = np.where(workflow.L==(level+1)%max_level) [0]
        task_dout =  workflow.D[dout, task] 
        lout = solution[dout]   #P('LOUT:', lout)
        eng_data_out = np.sum(task_dout) * (DE)
        #time_data_out_next =  [ task_dout[i] * DR[int(lout[i])] for i in range(len(dout)) ] 
        #time_data_out_final = [ task_dout[i] * DR[0] for i in range(len(dout)) ] if level==max_level-1  else  [ 0 for _ in range(len(dout)) ]

        #dx = np.where(workflow.L==(level)) [0]
        if level==max_level-1:
            time_data_out_final = [ task_dout[i] * DR[0] for i in range(len(dout)) ]
            time_data_out_next =  [ task_dout[i] * DR[int(lout[i])] for i in range(len(dout)) ] 
            assert( np.sum(np.abs(np.array(time_data_out_final)-np.array(time_data_out_next))) == 0 )
            # set on temp
            temp[dout, task] = time_data_out_next #P('changing-out', task, dout)
        
        #P('\t din{}, task_din:{}, lin:{}'.format(din, task_din, lin))
        #P('\t dout{}, task_dout:{}, '.format(dout, task_dout))
        #P('\ttask-size:{}'.format(task_size))
        #P('\t time_compute:{}, eng_compute:{}, '.format(time_compute, eng_compute))
        #P('\t time_data_in:{}, eng_data_in:{}, '.format(time_data_in, eng_data_in))
        #P('\t time_data_out_next:{}, eng_data_out:{}, '.format(time_data_out_next, eng_data_out))


        Energy += (eng_compute + eng_data_in + eng_data_out)

        #cA = [  time_data_in[0], eng_data_in, time_compute, eng_compute, eng_data_out, time_data_out_final[0] ]
        #print('COS_ARRA:',cA, 'sum:', np.sum(cA))

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # end of loop
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Delay = critical path cost
    temp[:,:] = np.transpose(temp)

    # prepare matrix for floyd-warshal shortest path (works with negative w8s)
    M = np.zeros((T+2,T+2))
    M[0:-1,0:-1] = temp[:,:]
    M[-1,:] = np.inf
    M[:,-1] = np.inf
    ml = np.where(workflow.L==max_level-1)[0]
    M[ml, -1] = M[ml, 0]
    M[ml, 0] = np.inf
    M[np.where(M==0.0)] += NZ_EPSILON
    lm = len(M)
    for i in range(lm):
        for j in range(i+1, lm):
            if M[i,j]!=np.inf:
                M[i,j] = -M[i,j]

    dist_matrix = floyd_warshall(csgraph=M, directed=True,return_predecessors=False)
    Delay = -1*np.min(dist_matrix[:,-1])
    total_cost = Energy + Delay
    #print('ENERGY, DELAY COST:',Energy, Delay  )
    return total_cost


def BASELINE(infra, workflow, NZ_EPSILON = 0.00001, return_costs=False):
    #sol = np.random.randint(0, infra.A, size=f.task_count()+1)
    #infra.randomize( probs=(0.5,0.5,0.5,0.5))
    T = workflow.task_count()
    POS_SOL =  infra.A**T
    costs=[]
    for i in range(POS_SOL):
        sol = np.zeros(T+1, dtype=np.int32)
        sol[0]=0
        sol[1:] = int2base(i, infra.A, T)
        c = COST(infra, workflow, sol, NZ_EPSILON = NZ_EPSILON)
        costs.append(c)
    min_cost, max_cost = np.min(costs), np.max(costs)
    min_costC, max_costC = np.where(costs==min_cost)[0], np.where(costs==max_cost)[0]
    if not return_costs:
        costs.clear()
    return POS_SOL, costs, min_cost, max_cost, min_costC, max_costC

