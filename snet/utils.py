import numpy as np
import datetime
import matplotlib.pyplot as plt
from .flow import BASELINE
from .basic import int2base, base2int
def baselines(vxps, verbose=0, P=print):

    for v,vxp in enumerate(vxps):
        possible, costs, min_cost, max_cost, min_cost_at, max_cost_at =  \
            BASELINE(vxp.env.infra, vxp.env.flowgen.fixed_flow, NZ_EPSILON = 0.00001, return_costs=(verbose>1) )
        vxp.env.flowgen.fixed_flow.min_cost, vxp.env.flowgen.fixed_flow.max_cost = min_cost, max_cost
        #<--- baseline verbose ------------------------------------------->#
        if verbose>0:
            P('\nValidation Flow: # {}'.format(v))
            #vxp.env.flowgen.fixed_flow.render(P=P)
            vxp.env.render(show_infra=True, P=P)
            P('Possible Solutions:', possible)
            P('Cost-Range:', str(min_cost)+", "+str(max_cost))
            P('Min-Cost Soultions:', min_cost_at)
            for mcs in min_cost_at:
                P(mcs, '\t', int2base(mcs, vxp.env.infra.A, vxp.env.flowgen.T))
            P('Max-Cost Soultions:', max_cost_at)
            for mcs in max_cost_at:
                P(mcs, '\t', int2base(mcs, vxp.env.infra.A, vxp.env.flowgen.T))
            if verbose>1:
                plt.figure(figsize=(6*vxp.env.flowgen.T,5))
                plt.plot(costs, marker='.', color='tab:blue', linewidth=0.7)
                plt.hlines(min_cost, 0, possible+1, color='tab:green', linewidth=0.7)
                plt.hlines(max_cost, 0, possible+1, color='tab:red', linewidth=0.7)
                plt.grid(axis='both')
                plt.show()
        #<--- baseline verbose ------------------------------------------->#


# - - - - - -  - - - - - - - -  - - - - - -  - - - - - - - -  - - - - - 

def validation(vxps, policy, verbose=1, P=print):
    res = []
    acts = []
    for v,vxp in enumerate(vxps):
        vxp.reset(clear_mem=True)
        if verbose>1:
            vxp.env.flowgen.fixed_flow.render()

        timesteps = vxp.explore(policy, None, moves=1, decay=vxp.NO_DECAY, episodic=True, test=True)
        _, rows, atrew, rows_aseq = vxp.summary(P=lambda *arg: None)

        row_reward = rows[0,4]
        row_cost = rows[0,6]
        row_eff = vxp.env.flowgen.fixed_flow.geteff(row_cost)
        res.append( ( row_reward, row_cost, row_eff ) )
        acts.append(rows_aseq)
    #end for
    
    if verbose>0:
        P('--------------------------------------')
        for i,(rew, cost, eff) in enumerate(res):
            sol = acts[i]
            P('\t[#]:{}\t[R]:{}\t[C]:{}\t[%]:{}\t[S]:{}'.format(i, rew, cost, round(eff*100,4), sol))
    return np.array(res)  #, np.array(acts)

    


def plot_results(refX, epsX, lrsX, bhsX, prefix, F):
    # plot results

    if type(refX)!=type(None):
        fig=plt.figure(figsize=(12,4))
        plt.plot(refX[:,1], linewidth=0.6, color='tab:green')
        plt.title('Efficiency')
        plt.ylim(0.90,1.01)
        plt.grid(axis='both')
        plt.show()
        F(fig,'eff_train_'+prefix)
        plt.close()

        # plot results
        fig=plt.figure(figsize=(12,4))
        plt.plot(refX[:,1], linewidth=0.6, color='tab:green')
        plt.title('Efficiency_scaled')
        plt.ylim(0.0,1.05)
        plt.grid(axis='both')
        plt.show()
        F(fig,'eff_train_scaled_'+prefix)
        plt.close()

        fig=plt.figure(figsize=(12,4))
        plt.plot(refX[:,0], linewidth=0.6, color='tab:blue')
        plt.title('Reward')
        plt.grid(axis='both')
        plt.show()
        F(fig,'rew_train_'+prefix)
        plt.close()

    if type(epsX)!=type(None):
        fig = plt.figure(figsize=(12,4))
        plt.plot(epsX, linewidth=0.6, color='tab:purple')
        plt.title('Epsilon')
        plt.grid(axis='both')
        plt.ylim(0,1)
        plt.show()
        F(fig,'eps_train_'+prefix)
        plt.close()

    if type(lrsX)!=type(None):
        fig = plt.figure(figsize=(12,4))
        plt.plot(lrsX, linewidth=0.6, color='tab:red')
        plt.title('Learn Rate')
        plt.grid(axis='both')
        plt.show()
        F(fig,'lrs_train_'+prefix)
        plt.close()

    if type(bhsX)!=type(None):
        fig = plt.figure(figsize=(12,4))
        plt.plot(bhsX, linewidth=0.6, color='tab:olive')
        plt.title('Batch Size')
        plt.grid(axis='both')
        plt.show()
        F(fig,'bhs_train_'+prefix)
        plt.close()