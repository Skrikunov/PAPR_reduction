import matplotlib.pyplot as plt
from scipy import signal
import system as sys
import numpy as np
import torch

def plot_res_allocation(PTX_allocation,RB_allocation,MOD_allocation,color_list,config):
    N_UE = config['N_UE']
    P_TX = config['P_TX']
    N_used = config['N_used']
    fontsize=12
    plt.figure(figsize=(13,N_UE*0.6))
    plt.title('Resourses allocation.'+
              'N_UE = ' + str(N_UE) +
              '; P_TX = ' + str(P_TX) +
              '; N_SC = ' + str(N_used),
              fontsize=16)
    plt.ylim(0,600)
    plt.xlim(0,1.75)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.xlabel('Real',fontsize=18)
    plt.ylabel('Subcarrier index',fontsize=20)
    
    SC = sys.GET_UE_SC_idx(torch.tensor(RB_allocation))
    plt.hlines(SC,0,0.3,'k',linewidths=1)
    plt.hlines(SC,0.3,1.75,'k','--',linewidths=1)
    plt.vlines(np.array([1,1.3,1.6,1.9])-0.65,0,600,'k','--',linewidths=1)

    for i in range(len(RB_allocation)):
        plt.fill_between(np.array([0,0.35]),SC[i],SC[i+1]+1,color=color_list[i])
        dy = 4
        dx = -0.6
#         text = str('P_UE = '+str(PTX_allocation[i])+
#                    ';   N_RB = '+str(RB_allocation[i])+
#                    ';   N_SC = '+str(12*RB_allocation[i])+
#                    ';   MOD = '+MOD_allocation[i]))
        
        plt.text(1.00+dx,(SC[i]+SC[i+1])/2-dy,'P_UE = '+str(PTX_allocation[i]),fontsize=fontsize)
        plt.text(1.3+dx,(SC[i]+SC[i+1])/2-dy,'N_RB = '+str(RB_allocation[i]),fontsize=fontsize)
        plt.text(1.6+dx,(SC[i]+SC[i+1])/2-dy,'N_SC = '+str(12*RB_allocation[i]),fontsize=fontsize)
        plt.text(1.9+dx,(SC[i]+SC[i+1])/2-dy,'MOD = '+MOD_allocation[i],fontsize=fontsize)
    return None


def filter_ccdf(ccdf):
    # replace close to zero elements with NaN (to delete from plot)
    close_to_zero = np.isclose(ccdf,0,atol=1e-8).astype('int')
    is_zero_idxs = np.nonzero(close_to_zero)
    ccdf[is_zero_idxs] = float('nan')
    return ccdf


def plot_CCDF(CCDF,LABELS,PAPR,figsize):
    plt.figure(figsize=figsize)
    plt.title("CCDF",fontsize=16)
    plt.xlabel("PAPR",fontsize=16)
    plt.ylabel("Probability",fontsize=16)
    
    plt.xlim(PAPR.min(),PAPR.max())
    plt.ylim(1e-8,1)
    
    points = (PAPR.max()-PAPR.min()).astype(int)
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


def plot_spectrum(SIGNALS,LABELS,TITLE,Fs,config):
    assert len(SIGNALS)==len(LABELS)
    fontsize=16
    N_fft=config['N_fft']
    # create window
    win = signal.get_window('hanning', N_fft)
    # create figure
    plt.figure(figsize=(12,8))
    plt.ylim(-40,20)

    PSD = []
    for s_t,lab in zip(SIGNALS,LABELS):
        S_temp = np.array(s_t.reshape(1,-1).cpu())
        f, Pxxf = signal.welch(S_temp, Fs, window=win, noverlap=N_fft//2, nfft=N_fft, return_onesided=False,scaling='density')
        plt.plot(f, pow2db(np.fft.ifftshift(Pxxf.T)),label=lab)
        PSD.append(Pxxf)
    plt.title(TITLE,fontsize=fontsize)
    plt.xlim(-Fs/2,Fs/2)
    plt.xlabel('Frequency,MHz',fontsize=fontsize)
    plt.ylabel('Power,dB',fontsize=fontsize)
    plt.grid()
    plt.legend(loc='upper right',fontsize=fontsize) 
    plt.show()
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