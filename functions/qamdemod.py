def QAM_demod(S_f,M,lin_complex_const,lin_decimal_const,ptx,unit_power=False):
    k=(600/1024)**0.5
    shape = S_f.shape
    diff = S_f.reshape(-1,1) - lin_complex_const.reshape(1,-1)*(ptx**0.5)/k/(lin_complex_const.abs()**2).sum()**0.5
    D = lin_decimal_const[torch.abs(diff).argmin(axis=1)].reshape(shape)
    return D

def DEMOD_signal(S_f,device,MOD_allocation,PTX_allocation,RB_allocation,constellations,config,info=False):
    """
    Generates OFDM signal with predefined mod/tx power/rb allocations
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
    except: print('QPSK constellation has not been loaded since it is not defined in config file.')
    try: QAM16_c,QAM16_d = constellations['QAM16_c'],constellations['QAM16_d']
    except: print('QAM16 constellation has not been loaded since it is not defined in config file.')
    try: QAM64_c,QAM64_d = constellations['QAM64_c'],constellations['QAM64_d']
    except: print('QAM64 constellation has not been loaded since it is not defined in config file.')
    try: QAM256_c,QAM256_d = constellations['QAM256_c'],constellations['QAM256_d']
    except: print('QAM256 constellation has not been loaded since it is not defined in config file.')
    try: QAM1024_c,QAM1024_d = constellations['QAM1024_c'],constellations['QAM1024_d']
    except: print('QAM1024 constellation has not been loaded since it is not defined in config file.')
        
    PTX_allocation = torch.tensor(PTX_allocation)
    RB_allocation = torch.tensor(RB_allocation)
    UE_SC_idx = GET_UE_SC_idx(RB_allocation)
#     k=(N_used/N_fft)**0.5
#     D = D.to(torch.complex64)
    # array for the f-domain signal
    D = torch.zeros([N_used,M],dtype=torch.float32,device=device)
    for user in range(N_UE):
        start_idx = UE_SC_idx[user]
        end_idx = UE_SC_idx[user+1]
        print(start_idx)
        UE_power = PTX_allocation[user]
        
        if MOD_allocation[user] == 'QPSK':
            points_qpsk = S_f[start_idx:end_idx,:]
            D[start_idx:end_idx,:] = QAM_demod(points_qpsk,M=4,lin_complex_const=QPSK_c,
                                                    lin_decimal_const=QPSK_d,ptx=UE_power,unit_power=True)
            
        
        elif MOD_allocation[user] == 'QAM16':
            points_qam16 = S_f[start_idx:end_idx,:]
            D[start_idx:end_idx,:] = QAM_demod(points_qam16,M=16,lin_complex_const=QAM16_c,
                                                    lin_decimal_const=QAM16_d,ptx=UE_power,unit_power=True)
        
        elif MOD_allocation[user] == 'QAM64':
            points_qam64 = S_f[start_idx:end_idx,:]
            D[start_idx:end_idx,:] = QAM_demod(points_qam64,M=64,lin_complex_const=QAM64_c,
                                                    lin_decimal_const=QAM64_d,ptx=UE_power,unit_power=True)
        
        elif MOD_allocation[user] == 'QAM256':
            points_qam256 = S_f[start_idx:end_idx,:]
            D[start_idx:end_idx,:] = QAM_demod(points_qam256,M=256,lin_complex_const=QAM256_c,
                                                    lin_decimal_const=QAM256_d,ptx=UE_power,unit_power=True)

        elif MOD_allocation[user] == 'QAM1024':
            points_qam1024 = S_f[start_idx:end_idx,:x]
            D[start_idx:end_idx,:] = QAM_demod(points_qam1024,M=1024,lin_complex_const=QAM1024_c,
                                                    lin_decimal_const=QAM1024_d,ptx=UE_power,unit_power=True)
#     # shift along FFT axis
#     S_f = torch.roll(S_f,N_zero//2,dims=0)
#     # IFFT along the 1st axis (vertical, freqs)
#     S_t = torch.fft.ifft(S_f, axis=0) * torch.sqrt(torch.tensor(N_fft))

#     if info:
#         print("The signal has been generated:")
#         print(f'Total power = {power:20.3f}')
#         print(f'OFDM symbols: {M:20.0f}')
#         print(f'IFFT length: {N_fft:20.0f}')
#         print(f'Mean power in freq dommain = {torch.mean(torch.abs(S_f)**2):.3f}')
#         print(f'Mean power in time dommain = {torch.mean(torch.abs(S_t)**2):.3f}')
        
    return D

def GET_UE_SC_idx(RB_allocation):
    """
    """
    return 12*torch.tensor([torch.sum(RB_allocation[:i]) for i in range(len(RB_allocation)+1)])

D_est = DEMOD_signal(S_f[ZERO_SHIFT:-ZERO_SHIFT,:],device,MOD_allocation,PTX_allocation,RB_allocation,constellations,cfg,info=False)