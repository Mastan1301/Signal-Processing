#!/usr/bin/env python3

# Decoder for BEC (binary erasure channel)

import numpy as np 
import matplotlib.pyplot as plt
import commpy.channels
import math

img = np.load('../binary_image.npy')
lx, ly = len(img), len(img[0])

code = np.loadtxt("../encoded_bits.dat")
code_len = len(code)
G, H = np.loadtxt("../G.dat"), np.loadtxt("../H.dat")
n, k = G.shape[1], G.shape[0]

img = np.array(img).flatten() #vectorising the matrix
size = len(img)

row = [[] for _ in range(len(H))] # for storing indices of non-zero entries (corresponding to a particular row or column) in H 
col = [[] for _ in range(n)]

for i in range(len(H)):
	for j in range(n):
		if(H[i][j] == 1):
			row[i].append(j)
			col[j].append(i)

wc, wr = len(col[0]), len(row[0])

# Utility functions
def count_e(arr): # function for counting the no. of erasure symbols in a code block
    return np.count_nonzero(arr == -1)

def xor(arr, n):
	res = 0
	for i in range(len(arr)):
		if i != n:
			res += arr[i]
	return res % 2

def assign(arr, colIndex):
    bit = 0
    for i in col[colIndex]:
        if arr[i] != -1:
            bit = arr[i]
            break
    
    for i in col[colIndex]:
        arr[i] = bit

    return arr

def belief_prop(bits):
    L = H.copy()
    res = np.zeros(math.ceil(k*len(bits)/n))

    for i in range(int(len(bits)/n)):
        r = bits[n*i : n*(i+1)]

        for j in range(len(L)):
            L[j] = np.dot(H[j], r)

        for _ in range(10): # no. of iterations
            # column operations
            for j in range(n):
                if count_e(L[:, j]) != wc:
                    L[:, j] = assign(L[:, j], j)
                    r[j] = L[0][j]

            # row operations
            for j in range(k):
                index = 0
                cnt = 0
                for l in range(n):
                    if L[j][l] == -1:
                        index = l
                        cnt += 1
                
                if cnt == 1:
                    L[j][index] = xor(L[j], index)

            res[k*i : k*(i+1)] = r[0:k]

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
plt.savefig('../figs/BEC.png')
plt.show() 