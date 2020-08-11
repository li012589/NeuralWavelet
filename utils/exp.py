import torch
import numpy as np

def measure(vector):
    return (vector**2).sum()

def expm(q,rtol=1e-3,maxStep=15):
    accumulator = torch.eye(q.shape[-1]).to(q)
    tmpq = q
    i = 1
    error = rtol*measure(q)
    while measure(tmpq) >= error:
        accumulator = tmpq +accumulator
        i+=1
        tmpq = torch.matmul(tmpq,q)/i
        if i>maxStep:
            break
    return accumulator

def expmv(q,v,rtol=1e-3,maxStep=15):
    accumulator = v
    tmpq = torch.matmul(q,v)
    i = 1
    error = rtol*measure(tmpq)
    while measure(tmpq) >= error:
        accumulator = tmpq + accumulator
        i+=1
        tmpq = torch.matmul(q,tmpq)/i
        if i > maxStep:
            break
    return accumulator


def logMinExp(a, b, epsilon=1e-8):
    """
    Computes the log of exp(a) - exp(b) in a (more) numerically stable fashion.
    Using:
     log(exp(a) - exp(b))
     c + log(exp(a-c) - exp(b-c))
     a + log(1 - exp(b-a))
    And note that we assume b < a always.
    """
    y = a + torch.log(1 - torch.exp(b - a) + epsilon)

    return y