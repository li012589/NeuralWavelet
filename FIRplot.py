import numpy as np
import torch


#h = np.array([1/4, 1/2, 1/4])

#h = np.array([1, 2, 1])

#h = np.array([-1/8, 1/4, 3/4, 1/4, -1/8])

#h = np.array([1/2, 1 /2])

h = np.array([-1/2, 1, -1/2])

def H_o(omega, h):
    sum = 0
    for n, term in enumerate(h):
        sum += term * np.exp(-1j * omega * n)
    return np.abs(sum)

step = 0.001

H = np.array([H_o(o * np.pi, h) for o in np.arange(0, 1 + step, step)])

from matplotlib import pyplot as plt

plt.plot(H)

plt.show()
