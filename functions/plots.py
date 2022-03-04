import matplotlib.pyplot as plt
from scipy import signal
import system as sys
import numpy as np
import torch


def plot_res_allocation(PTX_allocation,RB_allocation,MOD_allocation,color_list,config):
    N_UE = config['N_UE']
    N_RB = config['N_RB']
    P_TX = config['P_TX']
    N_used = config['N_used']
    fontsize=12
    plt.figure(figsize=(13,2))
    plt.title(f'Resourses allocation N_UE = {N_UE}, P_TX = {P_TX}, N_SC = {N_used}, N_RB = {N_RB}',fontsize=16)
    panel_len = 50
    plt.xlim(-panel_len,N_used)
    plt.ylim(0,1)
    plt.xticks(fontsize=14)
    plt.yticks([])
    
    # plt.xlabel('Real',fontsize=18)
    plt.xlabel('Subcarrier index',fontsize=20)
    
    SC = sys.GET_UE_SC_idx(torch.tensor(RB_allocation))
    plt.vlines(SC,0,1.0,'k',linewidths=1)
#     plt.vlines(SC,0.3,1.75,'k','--',linewidths=1)
    plt.hlines(np.array([0.2,0.4,0.6,0.8]),-panel_len,600,'k','--',linewidths=1)
    locs0 = np.array([0.2,0.4,0.6,0.8])
    locs1 = locs0 - 0.15

    for i in range(len(RB_allocation)):
        plt.fill_between(np.array([SC[i],SC[i+1]+1]),np.array([0.8]),np.array([1.0]),color=color_list[i])
        dy = 7
        dx = 9
        plt.text(-panel_len + 10,locs1[-1],f'MOD',fontsize=fontsize)
        plt.text(-panel_len + 10,locs1[-2],f'P_TX',fontsize=fontsize)
        plt.text(-panel_len + 10,locs1[-3],f'N_RB',fontsize=fontsize)
        plt.text(-panel_len + 10,locs1[-4],f'N_SC',fontsize=fontsize)
        
        plt.text((SC[i]+SC[i+1])/2 - 3.4*len(MOD_allocation[i]),locs1[-1],f'{MOD_allocation[i]}',fontsize=fontsize)
        plt.text((SC[i]+SC[i+1])/2 - 11,locs1[-2],f'{PTX_allocation[i]:0.1f}',fontsize=fontsize)
        plt.text((SC[i]+SC[i+1])/2 - 11,locs1[-3],f'{RB_allocation[i]}',fontsize=fontsize)
        plt.text((SC[i]+SC[i+1])/2 - 11,locs1[-4],f'{12*RB_allocation[i]}',fontsize=fontsize)
    return None


def filter_ccdf(ccdf):
    # replace close to zero elements with NaN (to delete from plot)
    close_to_zero = np.isclose(ccdf,0,atol=1e-8).astype('int')
    is_zero_idxs = np.nonzero(close_to_zero)
    ccdf[is_zero_idxs] = float('nan')
    return ccdf


def plot_CCDF(CCDF,LABELS,PAPR,figsize):
    plt.figure(figsize=figsize)
    plt.title("CCDF (Complementary Cumulative Distribution Function)",fontsize=16)
    plt.xlabel("PAPR",fontsize=16)
    plt.ylabel("Probability",fontsize=16)
    
    plt.xlim(PAPR.min(),PAPR.max())
    plt.ylim(1e-6,1)
    
    points = (PAPR.max()-PAPR.min()).to(int)
    plt.xticks(np.linspace(0,len(PAPR),points+1),np.round(np.arange(PAPR.min(),PAPR.max()+1,1),2),fontsize=12)
    plt.yticks(fontsize=12)
    for ccdf,label in zip(CCDF,LABELS):
        # plot graph
        plt.semilogy(filter_ccdf(ccdf),label=label)
    plt.legend(loc='upper right',fontsize=14)
    plt.grid()
    plt.show()
    return None


def plot_allocations(RB_allocation,PTX_allocation,ANL_allocation):
    XLABELS=['UE number','UE number','UE number','UE number']
    YLABELS=['RB amount','TX power','Noise level','Contribution']
    TITLES=['RB_allocation','PTX_allocation','ANL_allocation','Contribution']
    VALUES=[RB_allocation,PTX_allocation,ANL_allocation,RB_allocation*ANL_allocation**0.5]
    color_list = np.random.rand(len(RB_allocation),3)
    fontsize=16
    assert len(XLABELS)==len(YLABELS)==len(TITLES)==len(VALUES)
    assert len(RB_allocation)==len(PTX_allocation)==len(ANL_allocation)==len(color_list)

    f, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2, sharey=False,figsize=(11,6))
    f.tight_layout(w_pad = 4.0, h_pad = 4.0)
    for ax,val,title,xlab,ylab in zip((ax1, ax2, ax3, ax4),VALUES,TITLES,XLABELS,YLABELS):
        ax.bar(np.arange(len(RB_allocation))+1,val,color=color_list)
        ax.set_title(title,fontsize=fontsize)
        ax.set_ylim(0,1.25*np.max(val)) 
        ax.set_xlabel(xlab,fontsize=fontsize)
        ax.set_ylabel(ylab,fontsize=fontsize)
    return color_list


def pow2db(val):
    return 10*np.log10(val)


def plot_spectrum(SIGNALS,LABELS,TITLE,ANL_allocation,SC,config,figsize):
    assert len(SIGNALS)==len(LABELS)
    fontsize=16
    N_fft=config['N_fft']
    Fs=config['Fs']
    N_UE=config['N_UE']
    # create window
    win = signal.get_window('hanning', N_fft)
#     win = np.ones(N_fft)
    # create figure
    plt.figure(figsize=figsize)
    plt.ylim(-90,20)

    PSD = []
    for s_t,lab in zip(SIGNALS,LABELS):
        S_temp = np.array(s_t.T.reshape(1,-1).cpu())
        f, Pxxf = signal.welch(S_temp, Fs, window=win, noverlap=N_fft//2, nfft=N_fft, return_onesided=False,scaling='density')
        plt.plot(f, pow2db(np.fft.ifftshift(Pxxf.T)),label=lab)
        PSD.append(Pxxf)
        
    for i in range(N_UE):
        allowed_level = 10*np.log10(ANL_allocation[i])
        top_level = 10*np.log10(np.array(PSD[0])[0,SC[i]:SC[i+1]]).mean()
        if i == 0: plt.hlines(top_level + allowed_level,(Fs/1024)*SC[i]-Fs/2,(Fs/1024)*SC[i+1]-Fs/2,colors='r',linewidth=2,label='Allowed noise level')
        else: plt.hlines(top_level + allowed_level,(Fs/1024)*SC[i]-Fs/2,(Fs/1024)*SC[i+1]-Fs/2,colors='r',linewidth=2)
#         plt.hlines(top_level + allowed_level,-Fs/2,Fs/2,colors='k',linestyles='--',linewidth=1)
        
    plt.title(TITLE,fontsize=fontsize)
    plt.xlim(-Fs/2,Fs/2)
    plt.xlabel('Frequency,MHz',fontsize=fontsize)
    plt.ylabel('Power,dB',fontsize=fontsize)
    plt.grid()
    plt.legend(loc='upper right',fontsize=12) 
    plt.show()
    
#     plt.figure(figsize=figsize)
#     plt.plot(-10*np.log10(PSD[0].T) + 10*np.log10(PSD[1].T))
#     for i in range(N_UE):
#         plt.hlines(10*np.log10(ANL_allocation[i]),SC[i],SC[i+1],colors='r',linewidth=3)
#         plt.hlines(10*np.log10(ANL_allocation[i]),0,1024,colors='k',linestyles='--',linewidth=1)
#     plt.xlim(0,1024)
#     plt.title(TITLE,fontsize=fontsize)
#     plt.xlabel('Frequency,MHz',fontsize=fontsize)
#     plt.ylabel('Power difference,dB',fontsize=fontsize)
#     plt.grid()
# #     plt.legend(loc='upper right',fontsize=fontsize) 
#     plt.show()
    
    return PSD


def plot_constellation(linear_complex_constellation,dec_num_linear_constellation,device):
    assert len(linear_complex_constellation) == len(dec_num_linear_constellation)
    if device==torch.device('cuda'):
        linear_complex_constellation=np.array(linear_complex_constellation.cpu())
        dec_num_linear_constellation=np.array(dec_num_linear_constellation.cpu()).astype(int)
    elif device==torch.device('cpu'):
        linear_complex_constellation=np.array(linear_complex_constellation)
        dec_num_linear_constellation=np.array(dec_num_linear_constellation).astype(int)
        
    M = len(linear_complex_constellation)
    k = int(np.log2(M))
    size = np.max(np.abs(linear_complex_constellation.real))*1.2

    if M==4:
        plt.figure(figsize=(8,8))
        fs = 18
        dx1=-0.02
        dx2=-0.01
        dy1=+0.02
        dy2=-0.05
    elif M==16:
        plt.figure(figsize=(8,8))
        fs = 18
        dx1=-0.02
        dx2=-0.01
        dy1=+0.02
        dy2=-0.05
    elif M==64:
        plt.figure(figsize=(12,12))
        fs = 14
        dx1=-0.012
        dx2=-0.005
        dy1=+0.005
        dy2=-0.01
    elif M==256:
        plt.figure(figsize=(22,22))
        fs = 12
        dx1=-1
        dx2=-0.0
        dy1=+0.5
        dy2=-0.75

    plt.scatter(0,0)

    plt.xlim(-size,size)
    plt.ylim(-size,size)
    plt.xlabel('Real',fontsize=24)
    plt.ylabel('Imaginary',fontsize=24)
    plt.vlines(0,-size,size,'k',linewidth=0.5)
    plt.hlines(0,-size,size,'k',linewidth=0.5)

    for i,point in enumerate(linear_complex_constellation):
        plt.scatter(point.real, point.imag)
        plt.text(point.real+dx1, point.imag+dy1,np.binary_repr(dec_num_linear_constellation.reshape(-1,1)[i].item(),width=k),fontsize=fs)
        plt.text(point.real+dx2, point.imag+dy2,str(dec_num_linear_constellation.reshape(-1,1)[i].item()),fontsize=fs)
    plt.show()


def find_max_min_papr_symbol(S_t,S_f,PAPR):
    min_papr,max_papr = {},{}
    min_papr['index'] = torch.nonzero(PAPR == torch.min(PAPR)).item()
    min_papr['value'] = PAPR[min_papr['index']].item()
    min_papr['symbol_t'] = S_t[:,min_papr['index']].cpu().squeeze()
    min_papr['symbol_f'] = S_f[:,min_papr['index']].cpu().squeeze()
    max_papr['index'] = torch.nonzero(PAPR == torch.max(PAPR))
    max_papr['value'] = PAPR[max_papr['index']].item()
    max_papr['symbol_t'] = S_t[:,max_papr['index']].cpu().squeeze()
    max_papr['symbol_f'] = S_f[:,max_papr['index']].cpu().squeeze()
    return min_papr,max_papr


def plot_maxminpapr(min_papr,max_papr,figsize):
    fontsize = 16
    fig, (ax1, ax2) = plt.subplots(2, 1, sharey=False,figsize=figsize)
    s = torch.max(torch.max(torch.abs(max_papr['symbol_t'])**2),torch.max(torch.abs(max_papr['symbol_t'])**2))
    ax1.set_xlim(0,len(max_papr['symbol_t']))
    ax1.set_ylim(0,1.05*s)
    ax1.set_ylabel('Power',fontsize=fontsize)
    ax1.set_xlabel('Time index',fontsize=fontsize)
    ax1.plot(torch.abs(min_papr['symbol_t'])**2,label='power')
#     ax1.stem(torch.abs(min_papr['symbol_t'])**2,label='power',markerfmt='')
    ax1.set_title(f'Min PAPR symbol has PAPR = {min_papr["value"]:1.3f}',fontsize=fontsize)
    ax1.grid()
    ax1.hlines(torch.mean(torch.abs(min_papr['symbol_t'])**2),0,1024,'r','--',linewidth=3,label='Mean power')
    ax1.legend(loc='upper right',fontsize=fontsize)
    
    ax2.set_xlim(0,len(min_papr['symbol_t']))
    ax2.set_ylim(0,1.05*s)
    ax2.set_ylabel('Power',fontsize=fontsize)
    ax2.set_xlabel('Time index',fontsize=fontsize)
    ax2.plot(torch.abs(max_papr['symbol_t'])**2,label='power')
#     ax2.stem(torch.abs(max_papr['symbol_t'])**2,label='power',markerfmt='')
    ax2.set_title(f'Max PAPR symbol has PAPR = {max_papr["value"]:1.3f}',fontsize=fontsize)
    ax2.grid()
    ax2.hlines(torch.mean(torch.abs(max_papr['symbol_t'])**2),0,1024,'r','--',linewidth=3,label='Mean power')
    ax2.legend(loc='upper right',fontsize=fontsize) 
    
    fig.tight_layout()
    plt.show()