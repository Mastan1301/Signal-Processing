#!/usr/bin/env python3

# Decoder for BEC (binary erasure channel)

import numpy as np 
import matplotlib.pyplot as plt
import commpy.channels

img = np.load('../binary_image.npy')
lx, ly = len(img), len(img[0])

code = np.loadtxt("../encoded_bits.dat")
code_len = len(code)
G, H = np.loadtxt("../G.dat"), np.loadtxt("../H.dat")
n, k = G.shape[1], G.shape[0]

img = np.array(img).flatten() #vectorising the matrix
size = len(img)

def belief_prop(r):
    res = np.zeros(size)
    return res



P = [0.1, 0.2, 0.3, 0.4, 0.5] # probability of erasure
BER = []

for i in P:
    print("For p =", i)
    r = commpy.channels.bec(code, i) # received bits. Here, '-1' is the erasure symbol.
    decod = belief_prop(r)
    error = (img != decod).sum()
    print("No. of incorrectly decoded bits:", error)
    BER.append(error/size)
    print("Bit Error rate:", error/size, "\n")

plt.plot(P, BER)
plt.xlabel('p')
plt.ylabel('BER')
#plt.savefig('BEC.jpg')
plt.show() 