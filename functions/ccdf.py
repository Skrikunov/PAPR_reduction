import torch

def get_CCDF(papr_range,S_t):
    """
    Сalculates CCDF of an input signal for a given PAPR range
    papr_range: np.linspace(start,stop,steps)
    S_t: np.array(n_OFDM_symbols,N_fft)
    """
    # find power for each sample
    P = torch.abs(S_t)**2
    # find PAPR (in times)
    P_ratio = P/torch.mean(P)
    # find PAPR (in dB)
    P_dB = 10*torch.log10(P_ratio)
    # CCDF PAPR zero array
    CCDF = torch.zeros_like(papr_range)
    for i,papr in enumerate(papr_range):
        CCDF[i] = torch.sum(P_dB >= papr)
    CCDF /= torch.numel(S_t)
    return CCDF