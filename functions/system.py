import numpy as np
import torch
import qammod as qmd


def PTX_allocate(config,info=False):
    """
    Allocates transmitted power for each systems' UE
        config - setup settings
        info - print function info (True/False)
    """
    # unpack configs
    P_TX = config['P_TX']
    N_UE = config['N_UE']
    P_UE_max_min = config['P_UE_max_min']
    SEED = config['P_SEED']
    if SEED:
#         torch.manual_seed(SEED)
        np.random.seed(SEED)
    
    # UE power constrains
    P_UE_min = P_TX / (N_UE - 1 + P_UE_max_min)
    P_UE_max= P_UE_max_min * P_UE_min

    # uniform power distribution
    P_allocation = np.round(np.random.random(N_UE)*P_TX)
    P_allocation = P_allocation + P_TX//10
    P_allocation = np.round(P_TX * P_allocation/np.sum(P_allocation))
    P_sum = np.sum(P_allocation)
    
    if P_sum != P_TX:
        delta_p = P_TX - P_sum
        random_UE = np.random.randint(N_UE)
        P_allocation[random_UE] += delta_p
        
    if info:
        print('* UE power has been allocated - OK:')
        print(f'Max UE power can be: {P_UE_max:10.3f}')
        print(f'Min UE power can be: {P_UE_min:10.3f}')
        print(f'Sum UE power can be: {P_UE_max+(N_UE-1)*P_UE_min:10.3f}')
        print(f'Current Ptx allocation: {P_allocation}',f'Sum = {np.sum(P_allocation):0.3f}')
        print()
        
    assert (np.sum(P_allocation <= 0) == 0), 'Power must be non-zero positive value'
    return P_allocation


def MOD_allocate(config,info=False):
    """
    Allocates modulation type for each systems' UE
        config - setup settings
        info - print function info (True/False)
    """
    modulations = {}
    modulations['QPSK']=config['EVM_QPSK']
    modulations['QAM16']=config['EVM_QAM16']
    modulations['QAM64']=config['EVM_QAM64']
    modulations['QAM256']=config['EVM_QAM256']
    modulations['QAM1024']=config['EVM_QAM1024']
    N_UE = config['N_UE']
    SEED = config['M_SEED']
    if SEED:
#         torch.manual_seed(SEED)
        np.random.seed(SEED)
    
    # an amount of modulation types
    mod_number = len(modulations.keys())
    # the list of modulation types and...
    mod_types = [mod_type for mod_type in modulations.keys()]
    # it's allowed EVM level (%)
    EVM_allowed = np.zeros(mod_number)
    for i,key in enumerate(modulations):
        EVM_allowed[i] = modulations[key]
        
    allowed_noise = np.zeros(N_UE)
    # random modulation index allocation
    mod_idx = np.random.randint(0, mod_number, size = N_UE)
    # allocate EVM
    EVM_allocation = EVM_allowed[mod_idx]
    # allocate names
    MOD_allocation = [mod_types[i] for i in mod_idx]
    
    if info:
        print('* MODs have been allocated - OK:')
        print('Current MODs allocation:',MOD_allocation)
        print('Current EVMs allocation:',EVM_allocation)
        print()
    return MOD_allocation,EVM_allocation


def RB_allocate(config,info=False):
    """
    Allocates resourse blocks for each systems' UE
        config - setup settings
        info - print function info (True/False)
    """
    N_UE = config['N_UE']
    N_RB = config['N_RB']
    SEED = config['RB_SEED']
    if SEED:
#         torch.manual_seed(SEED)
        np.random.seed(SEED)
    
    rb_alloc = np.round(np.random.random(N_UE)*N_RB) + 4 
    rb_alloc = rb_alloc + N_RB//3
    rb_alloc = np.round(N_RB * rb_alloc/np.sum(rb_alloc))
    rb_sum = np.sum(rb_alloc)
    
    if rb_sum != N_RB:
        delta_rb = N_RB - rb_sum
        random_UE = np.random.randint(N_UE)
        rb_alloc[random_UE] += delta_rb
    rb_alloc=rb_alloc.astype(int)
    
    if info:
        print('* RBs have been allocated - OK:')
        print('Current RB allocation:',rb_alloc,'Sum =',np.sum(rb_alloc))
        print()
        
    assert (np.sum(rb_alloc <= 0) == 0), 'Resourse block must be integer non-zero positive value'
    return rb_alloc


def get_ANL_allocation(EVM_allocation,PTX_allocation,info=False):
    """
    Returns allowed noise level allocation in terms of power and amplitude
        EVM_allocation - UEs EVM corresponding to modulation type
        PTX_allocation - allocated UEs powers
        info - print function info (True/False)
    """
    ANL_allocation_P = ((EVM_allocation/100)**2)*(PTX_allocation/PTX_allocation.sum()) # power
#     if info:
#         print(f'Allowed noise level allocation (dB): {10*np.log10(ANL_allocation_P)}')
#         print(f'(%): {100*ANL_allocation_P}')
#         print()

    ANL_allocation_A = (EVM_allocation/100) * (PTX_allocation/PTX_allocation.sum())**0.5  # amplitude
    if info:
        print(f'Allowed noise level allocation: {np.round(20*np.log10(ANL_allocation_A),3)} dB')
        print(f'                                {np.round(100*ANL_allocation_A,3)} %')
        print()
    return ANL_allocation_P,ANL_allocation_A


def find_peaks(S_t, peak_th, N_fft):
    """
    Finds peaks over predefined threshold
        S_t - time domain signal (torch tensor)
        th - peak selection threshold (torch tensor)
        N_fft - FFT length (torch tensor)
    """
    # compute signal power for each sample
    power = torch.abs(S_t)**2
    # find mean power for each OFDM symbol
    power_mean = torch.sum(power,axis=0) / N_fft
    # pass peaks higher than threshold
    # only th > 0 has sense (to substrant peaks higher mean value)
    S_t_peaks = S_t*(power > power_mean.reshape(1,-1)*10**(peak_th/10))
    # power_max,_ = torch.max(power,axis=0)
    # S_t_peaks = S_t*(power > power_max.reshape(1,-1)*peak_th)
    return S_t_peaks


def find_peaks1(S_t, n_peaks):
    Peaks_pos = torch.abs(S_t).argsort(axis=0)[-n_peaks:,:]
    Peaks = torch.zeros_like(S_t)
    for i in range(S_t.shape[1]):
        peaks_pos = Peaks_pos[:,i]
        Peaks[peaks_pos,i] = S_t[peaks_pos,i]
    return Peaks


def MOD_signal(D,device,MOD_allocation,PTX_allocation,RB_allocation,constellations,config,info=False):
    """
    Modulates OFDM signal with predefined mod/tx power/rb allocations
        D - decimal complex constelllation points
        device - allocate to torch.device('device') ('cpu'/'cuda')
        MOD_allocation - modulation typ allocation
        PTX_allocation - transmitted power allocation
        RB_allocation - resourse block allocation
        constellations - the set of consteaaltions (decimal + complex)
        config - setup settings
        info - print function info (True/False)
    """
    # unpack config
    N_UE = config['N_UE']
    M = config['M']
    N_fft = config['N_fft']
    N_used = config['N_used']
    N_zero = config['N_zero']
    N_SC_RB = config['N_SC_RB']
    power=0
    # get complex and decimal constellations
    try: QPSK_c,QPSK_d = constellations['QPSK_c'],constellations['QPSK_d']
    except: None # print('QPSK constellation has not been loaded since it is not defined in config file.')
    try: QAM16_c,QAM16_d = constellations['QAM16_c'],constellations['QAM16_d']
    except: None # print('QAM16 constellation has not been loaded since it is not defined in config file.')
    try: QAM64_c,QAM64_d = constellations['QAM64_c'],constellations['QAM64_d']
    except: None # print('QAM64 constellation has not been loaded since it is not defined in config file.')
    try: QAM256_c,QAM256_d = constellations['QAM256_c'],constellations['QAM256_d']
    except: None # print('QAM256 constellation has not been loaded since it is not defined in config file.')
    try: QAM1024_c,QAM1024_d = constellations['QAM1024_c'],constellations['QAM1024_d']
    except: None # print('QAM1024 constellation has not been loaded since it is not defined in config file.')
        
    PTX_allocation = torch.tensor(PTX_allocation)
    RB_allocation = torch.tensor(RB_allocation)
    UE_SC_idx = GET_UE_SC_idx(RB_allocation)
    D = D.to(torch.complex64)
    # array for the f-domain signal
    S_f = torch.zeros([N_fft,M],dtype=torch.complex64,device=device)
    for user in range(N_UE):
        UE_power = torch.sqrt(PTX_allocation[user])
        start_idx = UE_SC_idx[user]
        end_idx = UE_SC_idx[user+1]
        N_SC=N_SC_RB*RB_allocation[user] 
        
        k=1
#         k=(N_SC/N_fft)**0.5
        k=(N_used/N_fft)**0.5

        
        if MOD_allocation[user] == 'QPSK':
            data_qpsk = D[start_idx:end_idx,:] # torch.randint(0,4,(M,N_SC),device=device)
            S_f[start_idx:end_idx,:] = qmd.QAM_mod(data_qpsk,M=4,lin_complex_const=QPSK_c,
                                                    lin_decimal_const=QPSK_d,unit_power=True)*UE_power/k
        
        elif MOD_allocation[user] == 'QAM16':
            data_qam16 = D[start_idx:end_idx,:] # torch.randint(0,16,(M,N_SC),device=device)
            S_f[start_idx:end_idx,:] = qmd.QAM_mod(data_qam16,M=16,lin_complex_const=QAM16_c,
                                                    lin_decimal_const=QAM16_d,unit_power=True)*UE_power/k
        
        elif MOD_allocation[user] == 'QAM64':
            data_qam64 = D[start_idx:end_idx,:] # torch.randint(0,64,(M,N_SC),device=device)
            S_f[start_idx:end_idx,:] = qmd.QAM_mod(data_qam64,M=64,lin_complex_const=QAM64_c,
                                                    lin_decimal_const=QAM64_d,unit_power=True)*UE_power/k
        
        elif MOD_allocation[user] == 'QAM256':
            data_qam256 = D[start_idx:end_idx,:] # torch.randint(0,256,(M,N_SC),device=device)
            S_f[start_idx:end_idx,:] = qmd.QAM_mod(data_qam256,M=256,lin_complex_const=QAM256_c,
                                                    lin_decimal_const=QAM256_d,unit_power=True)*UE_power/k

        elif MOD_allocation[user] == 'QAM1024':
            data_qam1024 = D[start_idx:end_idx,:] # torch.randint(0,1024,(M,N_SC),device=device)
            S_f[start_idx:end_idx,:] = qmd.QAM_mod(data_qam1024,M=1024,lin_complex_const=QAM1024_c,
                                                    lin_decimal_const=QAM1024_d,unit_power=True)*UE_power/k
            
        if info: power += get_power(S_f[start_idx:end_idx,:])
    
    # shift along FFT axis
    S_f = torch.roll(S_f,N_zero//2,dims=0)
    # IFFT along the 1st axis (vertical, freqs)
    S_t = torch.fft.ifft(S_f, axis=0) * torch.sqrt(torch.tensor(N_fft))

    if info:
        print("The signal has been generated:")
        print(f'Total power = {power:20.3f}')
        print(f'OFDM symbols: {M:20.0f}')
        print(f'IFFT length: {N_fft:20.0f}')
        print(f'Mean power in freq dommain = {torch.mean(torch.abs(S_f)**2):.3f}')
        print(f'Mean power in time dommain = {torch.mean(torch.abs(S_t)**2):.3f}')
        
    return S_t,S_f


def GEN_points(device,MOD_allocation,RB_allocation,config,info=False):
    """
    Generates OFDM signal with predefined mod/tx power/rb allocations
        device - allocate to torch.device('device') ('cpu'/'cuda')
        MOD_allocation - modulation typ allocation
        RB_allocation - resourse block allocation
        config - setup settings
        info - print function info (True/False)
    """
    # unpack config
    N_UE = config['N_UE']
    M = config['M']
    N_fft = config['N_fft']
    N_used = config['N_used']
    N_zero = config['N_zero']
    N_SC_RB = config['N_SC_RB']
    SEED = config['RNG_SEED']
    if SEED:
        torch.cuda.manual_seed(SEED)

    RB_allocation = torch.tensor(RB_allocation)
    UE_SC_idx = GET_UE_SC_idx(RB_allocation)

    D_points = torch.zeros([N_used,M],device=device)
    for user in range(N_UE):
        start_idx = UE_SC_idx[user]
        end_idx = UE_SC_idx[user+1]
        N_SC=N_SC_RB*RB_allocation[user]
        
        if MOD_allocation[user] == 'QPSK':
            data_qpsk = torch.randint(0,4,(N_SC,M),device=device)
            D_points[start_idx:end_idx,:] = data_qpsk
        
        elif MOD_allocation[user] == 'QAM16':
            data_qam16 = torch.randint(0,16,(N_SC,M),device=device)
            D_points[start_idx:end_idx,:] = data_qam16
        
        elif MOD_allocation[user] == 'QAM64':
            data_qam64 = torch.randint(0,64,(N_SC,M),device=device)
            D_points[start_idx:end_idx,:] = data_qam64
        
        elif MOD_allocation[user] == 'QAM256':
            data_qam256 = torch.randint(0,256,(N_SC,M),device=device)
            D_points[start_idx:end_idx,:] = data_qam256

        elif MOD_allocation[user] == 'QAM1024':
            data_qam1024 = torch.randint(0,1024,(N_SC,M),device=device)
            D_points[start_idx:end_idx,:] = data_qam1024
    return D_points


def GET_GROUP_SC(N_used,UE_indexes,RB_allocation):
    """
    Generates OFDM signal with predefined mod/tx power/rb allocations
        N_used - the number of used subcarriers
        UE_indexes - order number of users
        RB_allocation - allocation of resourseblocks for each UE
    """
    UE_SC_idx = GET_UE_SC_idx(RB_allocation)
    ALL_SC = np.arange(N_used)
    GROUP_UE_SC = np.array([]).astype(int)
    for UE_idx in UE_indexes:
        sc_current_UE = ALL_SC[UE_SC_idx[UE_idx]:UE_SC_idx[UE_idx+1]]
        GROUP_UE_SC = np.hstack((GROUP_UE_SC,sc_current_UE))
    return GROUP_UE_SC


def get_signal_PAPR(S_t):
    """
    Calculates peak-to-average power ratio of the given signal
        S_t - input signal
    """
    max_power = torch.max(torch.abs(S_t)**2,axis=0)[0]
    mean_power = torch.mean(torch.abs(S_t)**2,axis=0)
    PAPR = 10*torch.log10(max_power/mean_power)
    return PAPR


def GET_UE_RB_idx(RB_allocation):
    """
    """
    return np.array([np.sum(RB_allocation[:i]) for i in range(len(RB_allocation)+1)])


def GET_UE_SC_idx(RB_allocation):
    """
    """
    return 12*torch.tensor([torch.sum(RB_allocation[:i]) for i in range(len(RB_allocation)+1)])


def get_power(signal):
    """
    """
    return torch.mean(torch.abs(signal)**2)