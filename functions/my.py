import numpy as np
import torch
import torch.nn as nn
import papreduce as prd
import system as syst
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# THEORETICAL SPLITTING
def split_list(data, n):
    from itertools import combinations, chain
    for splits in combinations(range(1, len(data)), n-1):
        result = []
        prev = None
        for split in chain(splits, [None]):
            result.append(np.array(data[prev:split]))
            prev = split
        yield result

def get_splits(in_list,n_gr_max):
    groups = []
    for n_gr in range(1,n_gr_max+1):
        groups += list(split_list(in_list, n_gr))
    return groups



def opt_split_theor(g_num,ANL_allocation,RB_allocation): 
    """
    Compares all possible (and reasonable) splittings into groups theoretically (reduction ~ sqrt(ANL)*N_RB) and finds an optimal (the best) of them.
    """
    # sort UEs in descending order
    order = list(ANL_allocation.argsort()[::-1])
    groups = get_splits(order,g_num)

    contributions = []
    for i,split in enumerate(groups):
        G1=G2=G3=G4=None
        C1=C2=C3=C4=0
        try:
            G1 = split[0]
            C1 = np.sqrt(np.min(ANL_allocation[G1])) * RB_allocation[G1].sum()
        except: None
        try:
            G2 = split[1]
            C2 = np.sqrt(np.min(ANL_allocation[G2])) * RB_allocation[G2].sum()
        except: None
        try:
            G3 = split[2]
            C3 = np.sqrt(np.min(ANL_allocation[G3])) * RB_allocation[G3].sum()
        except: None
        try:
            G4 = split[3]
            C4 = np.sqrt(np.min(ANL_allocation[G4])) * RB_allocation[G4].sum()
        except: None
        reduction = C1+C2+C3+C4
        contributions.append(np.round(reduction,3))
        print(f'{i}) {split} {contributions[-1]}')

    idx_best = np.argmax(contributions)
    result = contributions[idx_best]
    split_best = groups[idx_best]
    print(f'BEST SPLIT: N = {idx_best},{split_best},result = {result}')
    return split_best,idx_best,result



# THEORETICAL SPLITTING K-Means
def kmeans_split(ANL_allocation,n_clusters):
    """
    Finds optima splitting into groups using k-means method.
    """
    X = np.array([[i] for i in ANL_allocation])
    kmeans = KMeans(n_clusters, random_state=0).fit(X)
    lables = kmeans.labels_

    G = [[] for _ in range(n_clusters)]
    for i in range(n_clusters):
        for j,item in enumerate(lables):
            if item == i:
                G[i].append(j)
    idxs = []
    for i,item in enumerate(G):
        idxs.append(np.min(ANL_allocation[G[i]]))
    order = np.argsort(idxs)[::-1]
    G = list(np.array(G,dtype=object)[order])

    for i,item in enumerate(G):
        G[i] = np.array(G[i])
    print('BEST SPLIT (KMEANS):',G)
    return G



# PASS THROUGH THE SYSTEM
class reduction_layer(torch.nn.Module):
    """
    PAPR reduction layer.
    """
    def __init__(self,peak_th,group_ANL_th,G_SC,S_sc,cfg):
        super(reduction_layer, self).__init__()
        self.peak_th = peak_th
        self.group_ANL_th = group_ANL_th
        self.G_SC = G_SC
        self.S_sc = S_sc
        self.cfg = cfg
        
    def forward(self, S_t):
        S_t_reduced = prd.PAPR_reduce(S_t,self.peak_th,self.group_ANL_th,self.G_SC,self.S_sc,self.cfg,info=False)
        return S_t_reduced

    def extra_repr(self):
        return f'peak_th={self.peak_th:0.3f},group_ANL_th={self.group_ANL_th:0.6f}'

        

def test_reduction(S_t,split_best,peak_ths,UE_ANLs,UE_SCs,S_sc,cfg,info):
    """
    Tests all reduction layers with different (or same) thresholds.
    """
    assert (len(split_best) == len(peak_ths)) or len(peak_ths) == 1, 'Group numbers and threshold numbers dont correspond each other! '

    g_num = len(split_best)
    if len(peak_ths) == g_num:
        OPTION = 'multiple thresholds'
    elif len(peak_ths) == 1:
        OPTION = 'single threshold'
        peak_ths = [peak_ths[0] for _ in range(g_num)]

    system = nn.Sequential()
    for i,(split,peak_th) in enumerate(zip(split_best,peak_ths)):
        system.add_module('layer_'+str(i),reduction_layer(peak_th,np.min(UE_ANLs[split]),np.concatenate([UE_SCs[j] for j in split]),S_sc,cfg))
    S_t_reduced = system.forward(S_t)

    PAPR0 = syst.get_signal_PAPR(S_t)
    max_papr_before = torch.max(PAPR0)
    PAPR = syst.get_signal_PAPR(S_t_reduced)
    max_papr_after = torch.max(PAPR)
    if info==True:
        print(f'MAX PAPR before: {max_papr_before.item():0.3f}')
        print(f'MAX PAPR after:  {max_papr_after.item():0.3f}')
        print(OPTION)
        print(f'Thresholds: {peak_ths}')
        print(system)
    return S_t_reduced



# PRACTICAL SPLITTING
def opt_split_pract(S_t,ANL_allocation,UE_ANLs,UE_SCs,S_sc,cfg):
    """
    Compares all possible (and reasonable) splittings into groups practically and finds an optimal (the best) of them.
    """
    g_num = 3
    desc_order = list(ANL_allocation.argsort()[::-1])
    groups = get_splits(desc_order,g_num)
    peak_ths = [9.]
    contributions = []
    for j,split in enumerate(groups):
        S_t_reduced = test_reduction(S_t,split,peak_ths,UE_ANLs,UE_SCs,S_sc,cfg,info=False)
        PAPR = syst.get_signal_PAPR(S_t_reduced)
        max_papr_current = torch.max(PAPR)
        contributions.append(np.round(max_papr_current.item(),3))
        print(f'{j}) {split} {contributions[-1]}')
            
    idx_best = np.argmin(contributions)
    max_papr_best = contributions[idx_best]
    split_best = groups[idx_best]
    print(f'BEST SPLIT: N = {idx_best},{split_best},result = {max_papr_best}')
    return split_best,idx_best,max_papr_best



def single_opt_th(S_t,th_range,split_best,UE_ANLs,UE_SCs,S_sc,cfg,info):
    """
    Finds a single optimal threshold for all groups.
    """
    def reduce(peak_th):
        system = nn.Sequential()
        for i,split in enumerate(split_best):
            system.add_module('layer_'+str(i),reduction_layer(peak_th,np.min(UE_ANLs[split]),np.concatenate([UE_SCs[j] for j in split]),S_sc,cfg))
        S_t_reduced = system.forward(S_t)
        PAPR = syst.get_signal_PAPR(S_t_reduced)
        max_papr = PAPR.max().item()
        return max_papr
        
    results = []
    # bisection
    bnd1 = np.array([th_range[0],th_range[-1]])
    for i in range(10):
        # find midddle point
        mid_point1 = (bnd1[0] + bnd1[1])/2
        mid_res1 = reduce(mid_point1)
        edg_res1 = [reduce(bnd1[0]),reduce(bnd1[1])]
        idx_del1 = np.array([edg_res1[0],edg_res1[1],mid_res1]).argmax()
        bnd1 = np.delete(np.array([bnd1[0],bnd1[1],mid_point1]),idx_del1)
        results.append(bnd1.mean().item())
    best_th = results[-1]
    if info:
        results1 = []
        for th in th_range:
            papr_max = reduce(th)
            results1.append(papr_max)
        plt.plot(th_range,results1)
        plt.vlines(results[-1],np.min(results1),np.max(results1),'r')
        best_idx = torch.argmin(torch.tensor(results1))
        print(f'PAPR after reduction:{reduce(best_th):0.3f}, best threshold: {best_th:0.3f}')
        print()
    return best_th



def find_opt_th(S_t,th_range,split,UE_ANLs,UE_SCs,S_sc,cfg):
    """
    """
    def reduce(peak_th):
        # reduce PAPR of input signal
        S_t_reduced = prd.PAPR_reduce(S_t,peak_th,np.min(UE_ANLs[split]),np.concatenate([UE_SCs[j] for j in split]),S_sc,cfg,info=False)
        # calculate max PAPR
        PAPR = syst.get_signal_PAPR(S_t_reduced)
        max_papr = PAPR.max().item()
        return max_papr
    
    # bisection method to find the minimum
    results = []
    bnd1 = np.array([th_range[0],th_range[-1]])
    for i in range(10):
        mid_point1 = (bnd1[0] + bnd1[1])/2
        mid_res1 = reduce(mid_point1)
        edg_res1 = [reduce(bnd1[0]),reduce(bnd1[1])]
        idx_del1 = np.array([edg_res1[0],edg_res1[1],mid_res1]).argmax()
        bnd1 = np.delete(np.array([bnd1[0],bnd1[1],mid_point1]),idx_del1)
        results.append(bnd1.mean().item())

    if 1:
        # grid search for plots
        results1 = []
        for th in th_range:
            papr_max = reduce(th)
            results1.append(papr_max)
        plt.plot(th_range,results1)
        plt.vlines(results[-1],np.min(results1),np.max(results1),'r')

    best_th = results[-1]
    print(best_th)
    return best_th

def multiple_opt_th(S_t,th_range,split,UE_ANLs,UE_SCs,S_sc,cfg,info):
    """
    Finds multiple optimal thresholds (for each PAPR reduction layer individually)
    """
    ths = [] # resulting thresholds
    # create a container
    system = nn.Sequential() 
    # find optimal thresholt for the 1st layer
    peak_th = single_opt_th(S_t,th_range,[split[0]],UE_ANLs,UE_SCs,S_sc,cfg,info=info)
    # save it
    ths.append(peak_th)
    # create a PAPR reduction layer
    system.add_module('layer_'+str(0),reduction_layer(peak_th,np.min(UE_ANLs[split[0]]),np.concatenate([UE_SCs[j] for j in split[0]]),S_sc,cfg))

    # for all layers
    for i in range(1,len(split)):
        # pass through existing system
        S_t_reduced = system.forward(S_t)
        # find optimal thresholt for the i-th layer
        peak_th = single_opt_th(S_t_reduced,th_range,[split[i]],UE_ANLs,UE_SCs,S_sc,cfg,info=info)
        # save it
        ths.append(peak_th)
        # create a PAPR reduction layer
        system.add_module('layer_'+str(i),reduction_layer(peak_th,np.min(UE_ANLs[split[i]]),np.concatenate([UE_SCs[j] for j in split[i]]),S_sc,cfg))
    # pass the signal via whole system
    S_t_reduced = system.forward(S_t)
    PAPR = syst.get_signal_PAPR(S_t_reduced)
    if info:
        print(f'PAPR after reduction:{PAPR.max():0.3f}, best thresholds: {ths}')
        print()
    return ths