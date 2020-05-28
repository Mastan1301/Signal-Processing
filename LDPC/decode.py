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

n, k = G.shape[1], G.shape[0]
M = 2

img = np.array(img).flatten() #vectorising the matrix
size = len(img)

def modulate(b1, Eb, i): # BPSK modulation function
    t = np.linspace((i-1)*T, i*T, int(T*f_s))
    res = np.zeros(len(t))
    if(b1 == 0):    
        for j in range(len(t)):        
            res[j] = np.cos(2*np.pi*f_c*t[j])
    
    if(b1 == 1):    
        for j in range(len(t)):
            res[j] = -np.cos(2*np.pi*f_c*t[j])

    return res

def WGN(variance, length): # function to generate white-gaussian-noise
    mu = 0
    res = np.zeros((length, n1))
    for i in range(length):
        res[i] = np.random.normal(mu, np.sqrt(variance), n1)
    return res

def decompose(x, Eb):
    cos = [np.cos(2*np.pi*f_c*i) for i in np.linspace(0, T, n1)] #basis function
    res = np.dot(x, cos)/len(x)
    return res

def second_min(vec, pos):
    minEl = 1e8
    for i in range(len(vec)):
        if (i != pos):
            minEl = min(minEl, vec[i])
    return minEl

def belief_prop(demod): # decoding using belief-propagation/sum-product/message-passing algorithm
    L = H.copy()
    decod = np.zeros(int(k*len(demod)/n)+1)

    for i in range(int(len(demod)/n)):
        r = demod[n*i : n*(i+1)]
        for j in range(len(L)):
                L[j] = np.dot(H[j], r) # equivalent to a Tanner graph construction

        prevDecision = np.zeros(n) # to store the previously obtained decision
        currDecision = np.zeros(n) # to store the current decision

        while True:
            for j in range(len(L)):
                # row operations
                x = L[j]
                pos = np.argmin(abs(x)) # using min-sum approximation to |log(tanh(|x|/2))| function
                m1 = abs(x[pos])
                m2 = second_min(abs(x), pos)
                S = np.product(np.sign(x))
                for l in range(len(x)):
                    if(l == pos):
                        x[l] = S * np.sign(x[l]) * m2
                    else:
                        x[l] = S * np.sign(x[l]) * m1

                L[j] = x

            # column operations
            for j in range(L.shape[1]):
                sum_j = r[j] + np.sum(L[:, j])
                L[:, j] = sum_j - L[:, j]
                if sum_j < 0:
                    currDecision[j] = 1
                else:
                    currDecision[j] = 0
            
            if(np.array_equal(currDecision, prevDecision)): # Stop the iteration if convergence is attained
                decod[k*i : k*(i+1)] = currDecision[0:k]
                break

            prevDecision = currDecision

    return decod
            

#Manually computed average energy using integration
Eg = 1e-6
E_avg = Eg/2

#Computing average energy per bit
Eb =(E_avg*n)/(np.log2(M)*k)
s = np.zeros((code_len, n1))
for i in range(code_len):
    s[i] = modulate(code[i], Eb, i) #modulating

var = 10
w = WGN(var, code_len)

r = s + w #received signal

r_sym = np.zeros(code_len)

for i in range(code_len):
    r_sym[i] = decompose(r[i], Eb) # converting from signal to symbols

decod = belief_prop(r_sym) #decoding 
error = (img != decod).sum()
decod = decod.reshape(lx, ly)
plt.imshow(decod, 'gray')
plt.show()
print("No. of incorrectly decoded bits:", error)
print("Bit Error rate:", error/size, "\n")


