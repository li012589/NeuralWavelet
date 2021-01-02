import numpy as np
import torch


#h = np.array([1/4, 1/2, 1/4])

#h = np.array([1, 2, 1])

#_h = np.array([-1/8, 1/4, 3/4, 1/4, -1/8])

#_h = np.array([1/2, 1 /2])

_h = np.array([-1/2, 1, -1/2])

h = np.zeros(32)

h[:_h.shape[0]] = _h
step = 0.001


from matplotlib import pyplot as plt


deltaexpp = np.exp((np.arange(h.shape[-1]) * 1j * step * np.pi))

modp = [np.exp((np.arange(h.shape[-1]) * 1j * 1 * -np.pi))]
for n in range(2 * int(1 / step)):
    _modp = modp[-1] * deltaexpp
    modp.append(_modp)

modp = np.vstack(modp)

Hhp = np.abs(modp.dot(h))

deltaexp = np.exp((np.arange(h.shape[-1]) * -1j * step * np.pi))

mod = [np.exp((np.arange(h.shape[-1]) * -1j * 1 * -np.pi))]
for n in range(2 * int(1 / step)):
    _mod = mod[-1] * deltaexp
    mod.append(_mod)

mod = np.vstack(mod)

Hh = np.abs(mod.dot(h))


from numpy.testing import assert_allclose
def H_o(omega, h):
    sum = 0
    for n, term in enumerate(h):
        sum += term * np.exp(-1j * omega * n)
    return np.abs(sum)


H = np.array([H_o(o * np.pi, h) for o in np.arange(0, 1 + step, step)])

plt.plot(H)
#assert_allclose(Hh, H)

plt.figure()

plt.plot(Hh)

plt.figure()

plt.plot(Hhp)

plt.show()
import pdb
pdb.set_trace()
