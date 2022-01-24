import datetime
now = datetime.datetime.now
import numpy as np
import matplotlib.pyplot as plt

#from core.params import MAX_N_TASKS

R = lambda x : round(x, 5)

def validation(validationS, policy, verbose=1, P=print):
    res = []
    acts = []
    for v,vxp in enumerate(validationS):
        vxp.reset(clear_mem=True)
        if verbose>1:
            vxp.env.flowgen.fixed_flow.render()

        timesteps = vxp.explore(policy, None, moves=1, epsilonF=lambda m: 0.0, episodic=True, test=True)
        _, rows, atrew, rows_aseq = vxp.summary(P=lambda *arg: None)

        row_reward = rows[0,4]
        row_cost = rows[0,6]
        #row_eff = vxp.env.flowgen.fixed_flow.geteff(row_cost)
        res.append( ( row_reward, row_cost ) )
        acts.append(rows_aseq)
    #end for
    
    if verbose>0:
        P('--------------------------------------')
        for i,(rew, cost) in enumerate(res):
            sol = acts[i]
            P('\t[#]:{}\t[R]:{}\t[C]:{}\t[S]:{}'.format(i, rew, cost, sol))
    return np.array(res)

def train_live_policy(explorer, policy, rolicy, path, validationS, paramS, reset_exp, P, F, S):
    do_val = not(len(validationS)==0)
    loss_h, val_h, param_h= \
        train_pie(explorer = explorer, 
                  policy = policy,  
                  rolicy=rolicy, 
                  validationS=validationS, 
                  params=paramS, 
                  reset_exp=reset_exp,
                  P=P ) 
    if do_val:
        plot_results(loss_h, val_h, param_h, F=F, S=S)
    _ = policy.save_external(path) if path!="" else None
    return loss_h, val_h, param_h

def train_pie(explorer, policy, rolicy, validationS, params, reset_exp, P=print):
    """
    .initial_lr
    .min_lr
    .lr_decay_freq
    .epochs
    .min_eps
    .explore_K
    .episodic_K
    .steps_K
    .batch_K
    .min_mem_K
    .test_freq
    """
    #P('[TRAIN_CALL]')
    #P(params.__dict__)
    stamp=now()
    do_val = not(len(validationS)==0)

    if do_val:
        res_val = validation(validationS, policy, verbose=1, P=P)
        P('PRE_TRAINING:', '\tReward:', np.sum(res_val[:, 0]),'\tCost:', np.sum(res_val[:, 1]))
    
    loss_hist, val_hist, param_hist = [], [], [] # [loss], [rew, cost], [eps, lr]
    
    
    if reset_exp:
        #for explorerX in explorerL:
        explorer.reset(clear_mem=True) #<--- reset()
    #explorer.env.render(show_infra=True, P=P)
    #for explorerX in explorerL:
    _ = explorer.explore(rolicy, None, 
                    params.min_mem_K, 
                    epsilonF=lambda m: 0.0, 
                    episodic=params.episodic_K)
    
    lr = params.initial_lr
    lr_decay_ratio =    (params.min_lr/params.initial_lr)**( 1/(params.epochs/params.lr_decay_freq)) 
    for epoch in range(params.epochs): 
    
        eps = max(params.min_eps, (1-(epoch/params.epochs))) #eps_hist.append(avg_eps)
        #for explorerX in explorerL:
        _ = explorer.explore(policy, rolicy, 
                        params.explore_K, 
                        epsilonF=lambda m: eps, 
                        episodic=params.episodic_K)
    
        if (epoch+1)%params.lr_decay_freq==0:
            lr*=lr_decay_ratio
        param_hist.append((eps, lr))
    
        # learn ------------------

        for _ in range(params.steps_K):
            #loss_arr = []
            #for explorerX in explorerL:
            avg_loss = policy.learn(
                explorer.memory,  
                explorer.memory.sample(params.batch_K * (explorer.env.T if params.episodic_K else 1)), 
                lr=lr, 
                dis=1)
            loss_hist.append(avg_loss)
        #loss_hist.append(loss_arr)
    
        # validate 
        
        if (epoch+1)%params.test_freq==0:
            if do_val:
                res = validation(validationS, policy, verbose=0, P=P)
                total_rew = np.sum(res[:, 0])
                total_cost = np.sum(res[:, 1])
                val_hist.append((total_rew, total_cost))
        
                P('[{}/{}]\t[R]:{}\t[C]:{}\t[L]:{}\t[lR]{}\t[E]:{}'.format(
                epoch+1, params.epochs, R(total_rew), R(total_cost), R(avg_loss), R(lr),  R(eps)))
            else:
                P('[{}/{}]\t[L]:{}\t[lR]{}\t[E]:{}'.format(
                epoch+1, params.epochs, R(avg_loss), R(lr),  R(eps)))
    
    
    P('Finished! Elapsed Time:', now()-stamp )
    
    loss_hist = np.array(loss_hist)
    val_hist = np.array(val_hist) # rew, cost
    param_hist = np.array(param_hist) # eps, lr
    
    # final testing
    if do_val:
        res_val = validation(validationS, policy, verbose=1, P=P)
        P('POST_TRAINING:', '\tReward:', np.sum(res_val[:, 0]),'\tCost:', np.sum(res_val[:, 1]))
    return loss_hist, val_hist, param_hist

def plot_results(loss_hist, val_hist, param_hist, F, S):
    
    data = [ 
        ('Loss',    loss_hist,      'tab:red',      'loss_h',   None),
        ('Reward',  val_hist[:,0],  'tab:blue',     'rew_h',   None),
        ('Cost',    val_hist[:,1],  'tab:green',    'cost_h',   None),
        ('Epsilon', param_hist[:,0],'tab:purple',   'eps_h',   (0,1)),
        ('LR',      param_hist[:,1],'tab:orange',   'lr_h',   None),
            ]
    lx = len(data)
    fig, ax = plt.subplots(lx,1, figsize=(16,lx*8))
    
    for i,( title, npy, col, npy_name, yl ) in enumerate(data):
        S(npy, npy_name)
        ax[i].plot(npy, color=col, linewidth=0.7)
        ax[i].set_ylabel(title)
        ax[i].grid(axis='both')
        ax[i].set_ylim(yl)
    plt.show()
    F(fig,'results')
    plt.close()
    








