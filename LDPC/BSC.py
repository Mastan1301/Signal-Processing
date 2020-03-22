# !/usr/bin/env python3

#Implementation of rate-1/4 regular LDPC code for a BSC

import numpy as np 
import matplotlib.pyplot as plt 
import random
from scipy.linalg import null_space

img = np.load('binary_image.npy')
lx, ly = len(img), len(img[0])
img = np.array(img).flatten()
size = lx*ly
n, k = 20, 5
m = n-k

def permut_index(s): #returns an array of randomly shuffled indices
    arr = list(range(s))
    res = []
    while len(arr):
        temp = random.choice(arr)
        res.append(temp)
        arr.remove(temp)
    return res

def pc_matrix(): #parity check matrix    
    H = np.zeros((m, n))
    wc, wr = 3, 4
    l1, l2 = int(n/wr), int(m/wc) # 5, 5

    for i in range(0, l2):
        for j in range(wr*i, wr*(i+1)):
            H[i][j] = 1

    for i in range(l2, m):
        if(i%l2 == 0):
            index_arr = permut_index(n)
        H[i, range(n)] = H[i%l2, index_arr]

    #print(H)
    return H

def gen_matrix(H): #generator matrix


H = pc_matrix()

        





