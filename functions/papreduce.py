import numpy as np
import torch
import qammod as qmd
import system as sys


def gen_fourier_matrix(config,device):
    """
    Generates Fourier matrix directly on the device
        config - setup settings
        device - allocate constellation to torch.device('device') ('cpu'/'cuda')
    """
    N_fft = config['N_fft']
    i = torch.arange(0,N_fft,device=device).reshape(-1,1).repeat(1, N_fft)
    t = torch.arange(0,N_fft,device=device).reshape(1,-1).repeat(N_fft,1)
    torch_pi = (2*torch.acos(torch.zeros(1)).item())
    S_sc = torch.zeros([N_fft,N_fft],dtype=torch.complex64,device=device)
    S_sc[:,:] = torch.exp( 2*torch_pi*1j*(i*t)/N_fft)
    return S_sc


def PAPR_reduce(S_t,peak_th,group_th,group_sc,S_sc,config,info=False):
    """
    Reduces PAPR of a given signal
        S_t - time domain input signal
        peak_th - peak slection threshold
        group_th - EVM threshold for current subcarriers set
        group_sc - subcarriers for current group
        S_sc - Fourier matrix
        config - setup settings
        info - print function info (True/False)
    """
    N_fft = config['N_fft']
    N_used = config['N_used']
    
    # find peaks
    Peaks = sys.find_peaks(S_t, peak_th, N_fft)
    
    # choose subcarriers in the Fourier matrix
    S_sc = S_sc[group_sc,:]
    
    # remove peaks on chosen subcarriers
    S_t_canc = (Peaks @ S_sc.conj().T / N_fft )@ S_sc
    
    # noise power per group subcarrier
    power_tones = torch.sum(torch.abs(S_t_canc)**2,axis=1)/group_sc.shape[0]
    # data power per used subcarrier
    power_data = torch.sum(torch.abs(S_t)**2,axis=1)/N_used
    
    # evaluate EVM
    u = -10*torch.log10(power_data/power_tones)

    # multiply by the coeff exceeding values
    th = 10*torch.log10(torch.tensor(group_th)) # arg - (EVM%**2/10000) * P/PTX
#     th = 20*torch.log10(group_th) # arg - (EVM%/100) * (P/PTX)**0.5
    
    coef = (u > th)*10**((-u+th)/20) + (u <= th)*1 # 0**((-u+th)/20)
    coef[torch.isnan(coef)] = 0
    S_t_canc_final = coef.reshape(-1,1) * S_t_canc
    
    if info:
        print("Pdata/Ptones before supression",np.round(-10*np.log10(np.array((power_data/power_tones).cpu())),2))
        power_tones1 = np.sum(np.abs(np.array(S_t_canc_final.cpu()))**2,axis=1)/group_sc.shape[0]
        print("Pdata/Ptones after supression",np.round(-10*np.log10(np.array((power_data.cpu()/power_tones1))),2))
    
    # reduce PAPR
    S_t_reduced = S_t - S_t_canc_final
    return S_t_reduced