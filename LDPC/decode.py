import numpy as np 
import matplotlib.pyplot as plt 

T =1e-6


def WGN(variance, length):
    s1, s2 = int(length/2), int(T*f_s)
    mu = 0
    res = np.zeros((s1, s2))
    for i in range(s1):
        res[i] = np.random.normal(mu, np.sqrt(variance), s2)
    return res

code = np.loadtxt("./encoded_bits.dat")
G, H = np.loadtxt("./G.dat"), np.loadtxt("./H.dat")

k, n = len(G), 1
M = 4
Eb_N0 = 1
E_avg = T
E_b =(E_avg*n)/(np.log2(M)*k)
