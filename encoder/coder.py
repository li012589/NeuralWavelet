import torch
import numpy as np

from . import rans

precision = 24

def encoder(CDF, symbol, state):

    def statfun_encode(s):
        return CDF[s], CDF[s + 1] - CDF[s]

    state = rans.append_symbol(statfun_encode, precision)(state, symbol)
    return state

def decoder(CDF, state):

    def statfun_decode(cdf):
        # Search such that CDF[s-1] <= cdf < CDF[s]
        s = np.searchsorted(CDF, cdf, side='right') - 1
        start = CDF[s]
        freq = CDF[s+1] - start
        return s, (start, freq)

    state, symbol = rans.pop_symbol(statfun_decode, precision)(state)
    return state, symbol