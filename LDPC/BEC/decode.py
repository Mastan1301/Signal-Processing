#!/usr/bin/env python3

# Decoder for BEC (binary erasure channel)

import numpy as np 
import matplotlib.pyplot as plt

img = np.load('../binary_image.npy')
lx, ly = len(img), len(img[0])

code = np.loadtxt("../encoded_bits.dat")
code_len = len(code)
G, H = np.loadtxt("../G.dat"), np.loadtxt("../H.dat")
n, k = G.shape[1], G.shape[0]

img = np.array(img).flatten() #vectorising the matrix
size = len(img)