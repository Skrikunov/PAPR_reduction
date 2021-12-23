import numpy as np
import torch 
import matplotlib.pyplot as plt


def constellation_power(constellation=None,info=False):
    """
    Calculates average power of an input constellation.
        constellation - an array of all complex constellation points (torch array)
        info - print function info (True/False)
    """
    power = torch.mean(torch.abs(constellation)**2)
    if info:
        print('Calculated power',"%.4f" % power0)
    return power


def constellation_normalize(constellation=None,info=False):
    """
    Constellation normalization to a unit power.
        constellation - an array of all complex constellation points (torch array)
        info - print function info (True/False)
    """
    power_before = constellation_power(constellation,info=False)
    power_after = constellation_power(constellation,info=False)
    constellation /= torch.sqrt(power_before)
    if info:
        print('Power before normalization',"%.4f" % power_before)
        print('Power after normalization',"%.4f" % power_after)
        print('Constellation has been normalized')
    return constellation


def gen_constellation(M=None,unit_power=False,gray_mapping=False,device=None):
    """
    Generates QAM constellation.
        M - constellation points number (4,16,64,256,1024)
        unit_power - scale constellation power to 1 (True/False)
        gray_mapping - use gray mapping (True/False)
        device - allocate constellation to torch.device('device') ('cpu'/'cuda')
    """
    assert (M==4 or M==16 or M==64 or M==256 or M==1024), "Incorrect modulation order. Only QPSK(QAM4,16,64,256,1024) are supported."
    
    # modulation order
    sqrt_M = np.sqrt(M).astype(int)
    # bit per complex symbol
    k=np.log2(M).astype(int)
    
    # mapping order
    mapping_order = np.arange(sqrt_M)
    if gray_mapping:
        # Binary to Gray code constelation convertor
        gray_order = np.bitwise_xor(mapping_order, mapping_order//2).astype(int)
        mapping_order = gray_order
    else:
        None
    
    # Gray code constelation to symbols convertor
    vect2 = np.arange(1, sqrt_M, 2)
    symbols = np.concatenate((np.flip(-vect2, axis=0), vect2)).astype(int)

    # decimal mapping
    dec_const_v = mapping_order[::-1].reshape(sqrt_M,1)
    dec_const_h = np.array([sqrt_M*i for i in range(sqrt_M)])[mapping_order].reshape(1,sqrt_M)
    dec_mapping = dec_const_v + dec_const_h

    # complex mapping
    const_v = 1j*symbols[::-1].reshape(sqrt_M,1)
    const_h = symbols.reshape(1,sqrt_M)
    complex_mapping = const_v + const_h

    # convert constellations into rows
    lin_complex_const = complex_mapping.reshape(-1,1)[:,0]
    lin_decimal_const = dec_mapping.reshape(-1,1)[:,0]
    
    # normalize constellation power
    if unit_power: lin_complex_const = constellation_normalize(torch.tensor(lin_complex_const),info=False)
    
    return torch.tensor(lin_complex_const,device=device),torch.tensor(lin_decimal_const,device=device)


def load_constellations(config=None,device=None,info=None):
    """
    Generates and loads QAM constellations in both decimal and complex forms.
        config - setup settings
        device - allocate constellation to torch.device('device') ('cpu'/'cuda')
    """
    unit_power=False
    gray_mapping=True
    constellations = {}
    loaded_list = []
    if 'EVM_QPSK' in config:
        QPSK_c,QPSK_d = gen_constellation(M=4,unit_power=unit_power,gray_mapping=gray_mapping,device=device)
        constellations['QPSK_c'],constellations['QPSK_d'] = QPSK_c,QPSK_d
        loaded_list.append('QPSK')
    if 'EVM_QAM16' in config:
        QAM16_c,QAM16_d = gen_constellation(M=16,unit_power=unit_power,gray_mapping=gray_mapping,device=device)
        constellations['QAM16_c'],constellations['QAM16_d'] = QAM16_c,QAM16_d
        loaded_list.append('QAM16')
    if 'EVM_QAM64' in config:  
        QAM64_c,QAM64_d = gen_constellation(M=64,unit_power=unit_power,gray_mapping=gray_mapping,device=device)
        constellations['QAM64_c'],constellations['QAM64_d'] = QAM64_c,QAM64_d
        loaded_list.append('QAM64')
    if 'EVM_QAM256' in config:
        QAM256_c,QAM256_d = gen_constellation(M=256,unit_power=unit_power,gray_mapping=gray_mapping,device=device)
        constellations['QAM256_c'],constellations['QAM256_d'] = QAM256_c,QAM256_d
        loaded_list.append('QAM256')
    if 'EVM_QAM1024' in config:
        QAM1024_c,QAM1024_d = gen_constellation(M=1024,unit_power=unit_power,gray_mapping=gray_mapping,device=device)
        constellations['QAM1024_c'],constellations['QAM1024_d'] = QAM1024_c,QAM1024_d
        loaded_list.append('QAM1024')
    if info: print('The following constelletions have been loaded:',loaded_list)
    return constellations


def QAM_mod(data,M,lin_complex_const,lin_decimal_const,unit_power=False):
    """
    Modulates input data in accordance with complex constellation.
        data - input data to be modulated (torch tensor)
        M - constellation points number (4,16,64,256,1024)
        lin_complex_const - linear array of complex constellation points
        lin_decimal_const - linear array of decimal constellation points
        unit_power - scale constellation power to 1 (True/False)
        gray_mapping - use gray mapping (True/False)
    """
    assert (M==4 or M==16 or M==64 or M==256 or M== 1024), "Incorrect modulation order. Only QPSK(QAM4),QAM16,QAM64,QAM256 are supported."
#     assert (torch.sum(data > M) == 0), "Incorrect data for current modulation order. Values should be in range [0,M)"
    
    # compare each data symbol with decimal constelletion (each element with all constellation)
    # find according index in the constellation
    idx_in_constellation = torch.argmax((data.reshape(-1,1) == lin_decimal_const.reshape(1,M)).int(),axis=1).reshape(data.shape)
    # modulated data (choose symbol from constellation for each data point)
    mod_data = lin_complex_const[idx_in_constellation]
    
    if unit_power:
        # don't use constellation_normalize() due to it's computation time
        mod_data /= torch.sqrt(torch.mean(torch.abs(mod_data)**2))
    else:
        None
        
    return mod_data