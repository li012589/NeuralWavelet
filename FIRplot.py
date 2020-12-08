import numpy as np
import torch


#h = np.array([1/4, 1/2, 1/4])

#h = np.array([1, 2, 1])

#h = np.array([-1/8, 1/4, 3/4, 1/4, -1/8])

#h = np.array([1/2, 1 /2])

_h = np.array([-1/2, 1, -1/2])

h = np.zeros(32)

h[:_h.shape[0]] = _h

def H_o(omega, h):
    sum = 0
    for n, term in enumerate(h):
        sum += term * np.exp(-1j * omega * n)
    return np.abs(sum)

step = 0.001

H = np.array([H_o(o * np.pi, h) for o in np.arange(0, 1 + step, step)])

from matplotlib import pyplot as plt

plt.plot(H)

deltaexp = np.exp((np.arange(h.shape[-1]) * 1j * step * np.pi))

mod = [np.exp((np.arange(h.shape[-1]) * 1j * 0 * np.pi))]
for n in range(int(1 / step)):
    _mod = mod[-1] * deltaexp
    mod.append(_mod)

mod = np.vstack(mod)

H = np.abs(mod.dot(h))

plt.figure()

plt.plot(H)
plt.show()
import pdb
pdb.set_trace()
