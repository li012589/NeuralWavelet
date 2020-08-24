import torch
import numpy as np

from . import rans

precision = 24

def encoder(CDF, symbol, state):
    '''
    Args:
        CDF: 1-D torch tensor, CDF[symbol] should return the CDF of the input symbol.
        symbol: uint8 for image or other type depending on the data to be encoded, symbol to be encoded.
        state: 1-D numpy.ndarray of type uint32 (or None if initialized), encoded data.

    Returns:
        state: 1-D numpy.ndarray of type uint32, encoded data.
    '''

    # numpy arrays are used to be compatible with rans.py
    CDF_prec = np.zeros(len(CDF)+1, dtype=np.uint32)
    for i in range(len(CDF)):
        CDF_prec[i + 1] = torch.round(CDF[i] * ((1 << precision) - 1))

    def statfun_encode(s):
        return CDF_prec[s], CDF_prec[s+1] - CDF_prec[s]

    if type(state) == type(None):
        state = rans.x_init
    else:
        state = rans.unflatten(state)

    state = rans.append_symbol(statfun_encode, precision)(state, symbol)
    state = rans.flatten(state)

    return state

def decoder(CDF, state):
    '''
    Args:
        CDF: 1-D torch tensor, CDF[symbol] should return the CDF of the input symbol.
        state: 1-D numpy.ndarray of type uint32, encoded data.

    Returns:
        state: 1-D numpy.ndarray of type uint32, encoded data.
        symbol: int64, decoded symbol.
    '''

    # numpy arrays are used to be compatible with rans.py
    CDF_prec = np.zeros(len(CDF)+1, dtype=np.uint32)
    for i in range(len(CDF)):
        CDF_prec[i + 1] = torch.round(CDF[i] * ((1 << precision) - 1))

    def statfun_decode(cdf):
        # Search such that CDF[s-1] <= cdf < CDF[s]
        s = np.searchsorted(CDF_prec, cdf, side='right') - 1
        start = CDF_prec[s]
        freq = CDF_prec[s+1] - start
        return s, (start, freq)

    state = rans.unflatten(state)
    state, symbol = rans.pop_symbol(statfun_decode, precision)(state)
    state = rans.flatten(state)

    return state, symbol