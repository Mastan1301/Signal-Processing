# SISO decoder for an AWGN channel using belief propagation/message passing algorithm.

import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import norm

T = 1e-6
f_s, f_c = 50e6, 2e6
n1 = int(T * f_s)
img = np.load('./binary_image.npy')
lx, ly = len(img), len(img[0])

code = np.loadtxt("./encoded_bits.dat")
code_len = len(code)
G, H = np.loadtxt("./G.dat"), np.loadtxt("./H.dat")

n, k = len(G), 1
M = 2

img = np.array(img).flatten() #vectorising the matrix
size = len(img)

def modulate(b1, Eb, i): # BPSK modulation function
    t = np.linspace((i-1)*T, i*T, int(T*f_s))
    res = np.zeros(len(t))
    if(b1 == 0):    
        for j in range(len(t)):        
            res[j] = np.sqrt(2*Eb/T) * np.cos(2*np.pi*f_c*t[j])
    
    if(b1 == 1):    
        for j in range(len(t)):
            res[j] = -np.sqrt(2*Eb/T) * np.cos(2*np.pi*f_c*t[j])

    return res

def WGN(variance, length): # function to generate white-gaussian-noise
    mu = 0
    res = np.zeros((length, n1))
    for i in range(length):
        res[i] = np.random.normal(mu, np.sqrt(variance), n1)
    return res

def decompose(x):
    cos = [np.cos(2*np.pi*f_c*i) for i in np.linspace(0, T, n1)] #basis function
    res = np.dot(x, cos)/(2*n1)
    return res

#def computeSign(vec):


def belief_prop(demod): # decoding using belief-propagation/sum-product algorithms
    L = H.copy()
    for i in range(int(len(demod)/n)):
        r = demod[i:n*i]
        for j in range(len(L)):
            L[j] = np.dot(L[j], r)
            # row operations
            x = L[j]
            pos = np.argmin(abs(x))
            m1 = x[pos]
            m2 = np.min(abs(x[x != m1]))
            S = np.product(np.sign(x))
            for k in range(len(x)):
                if(k == pos):
                    x[k] = S * np.sign(x[k]) * m2
                else:
                    x[k] = S * np.sign(x[k]) * m1
        


#Manually computed average energy using integration
E_avg = T

#Computing average energy per bit
Eb =(E_avg*n)/(np.log2(M)*k)
s = np.zeros((code_len, n1))
for i in range(code_len):
    s[i] = modulate(code[i], Eb, i) #modulating

#Here S_space entries are signal-space co-ordinates.
S_space = np.array([[-1, 1], [-1, -1], [1, -1], [1, 1]])
bit_array = np.array([[1, 0], [1, 1], [0, 1], [0, 0]])
M = len(S_space)

var = 10
w = WGN(var, code_len)

r = s + w #received signal

#demodulating scheme	
'''r_sym = np.zeros((int(code_len/2), 2))
for i in range(len(r_sym)):
    r_sym[i] = decompose(r[i]) #converting into signal space co-ordinates

dist = np.zeros((len(r_sym), M)) #matrix to store the distance values from the 4 symbols
for i in range(len(r)):
    for j in range(M):
        dist[i][j] = np.linalg.norm(r_sym[i] - S_space[j]) #calculating the distance

demod = np.zeros((code_len))
i = 0
while i < len(demod)/2:
    index = np.argmin(dist[i]) #minimum distance demodulation
    demod[2*i], demod[2*i+1] = bit_array[index][0], bit_array[index][1]
    i += 1'''

r_sym = np.zeros(code_len)
for i in range(code_len):
    r_sym[i] = decompose(r[i]) 

print(r_sym)
'''decod = belief_prop(demod) #decoding 
    
error = (img != decod).sum()
# decod = decod.reshape(lx, ly)
print("No. of incorrectly demodulated bits:", error)
print("Bit Error rate:", error/size, "\n")'''


