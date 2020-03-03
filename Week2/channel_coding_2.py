#EE18BTECH11039, EE18BTECH11010

#Implementation of rate-1/3 repetition channel coding and decoding for varying values of noise variance

import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import norm

#0 represents a black pixel and 1 is a white pixel.
T = 1e-6
f_s, f_c = 50e6, 2e6
n1 = int(T * f_s)

#The basis functions in the signal space representation.

img = np.load('mss.npy')
lx, ly = len(img), len(img[0])

img = np.array(img).flatten() #vectorising the matrix
size = len(img)

def bit(b):
    if b == 1:
        return -1
    return 1

def mod(b1, b2, i):
    t = np.linspace((i-1)*T, i*T, int(T*f_s))
    res = np.zeros(len(t))
    x1 = bit(b1)
    x2 = bit(b2)
    for j in range(len(t)):
        res[j] = (x1 * np.cos(2*np.pi*f_c*t[j])) + (x2 * np.sin(2*np.pi*f_c*t[j]))
    return res

def WGN(variance, length):
    s1, s2 = int(length/2), n1
    mu = 0
    res = np.zeros((s1, s2))
    for i in range(s1):
        res[i] = np.random.normal(mu, np.sqrt(variance), s2)
    return res

def decomp(x):
    cos = [np.cos(2*np.pi*f_c*i)*np.sqrt(2/T) for i in np.linspace(0, T, n1)] #basis functions
    sin = [np.sin(2*np.pi*f_c*i)*np.sqrt(2/T) for i in np.linspace(0, T, n1)]
    res = [np.dot(x, cos), np.dot(x, sin)]
    return res

def channel_code_2(x):
    code = np.zeros((3*len(x)))
    i = 0
    t = 0
    while t < len(x):
        for j in range(3):
            code[i+j] = x[t]
        t += 1
        i += 3
    return code

def channel_decode_2(y):
    decode = np.zeros((int(len(y)/3)))
    i, j = 0, 0
    while i < len(y):
        temp = y[i:i+3]
        if (temp == 0).sum() > 1:
            decode[j] = 0
        else:
            decode[j] = 1
        j += 1
        i += 3
    return decode

n, k = 3, 1
code_img = channel_code_2(img)
code_len = len(code_img)
s = np.zeros(((int(code_len/2)), int(T*f_s)))
for i in range(int(code_len/2)):
    s[i] = mod(code_img[2*i], code_img[2*i+1], i) #modulating

#Here S_space entries are signal-space co-ordinates.
S_space = np.array([[-1, 1], [-1, -1], [1, -1], [1, 1]])
bit_array = np.array([[1, 0], [1, 1], [0, 1], [0, 0]])
M = len(S_space)

#Manually computed average energy using integration
E_avg = T

#Computing average energy per bit
E_b =(E_avg*n)/(np.log2(M)*k)
var = [20, 12, 7, 5]
BER = []
Eb_N0 = []

print("For varying noise variance values....")
for t in range(len(var)):
	print("For variance =", var[t])
	w = WGN(var[t], code_len) #white gaussian noise
	N_0 = 2*var[t]/f_s
	r = s + w #received signal

	#demodulating scheme	
	r_sym = np.zeros((int(code_len/2), 2))
	for i in range(len(r_sym)):
	    r_sym[i] = decomp(r[i]) #converting into signal space co-ordinates

	dist = np.zeros((len(r_sym), M)) #matrix to store the distance values from the 4 symbols
	for i in range(len(r)):
	    for j in range(M):
		    dist[i][j] = np.linalg.norm(r_sym[i] - S_space[j]) #calculating the distance
    
	demod = np.zeros((len(code_img)))
	i = 0
	while i < len(demod)/2:
	    index = np.argmin(dist[i]) #minimum distance 
	    demod[2*i], demod[2*i+1] = bit_array[index][0], bit_array[index][1]
	    i += 1

	decod = channel_decode_2(demod)
        
	error = (img != decod).sum()
	decod = decod.reshape(lx, ly)
	print("No. of incorrectly demodulated bits:", error)
	BER.append(error/size)
	print("Bit Error rate:", error/size, "\n")

plt.semilogy(var, BER)
plt.xlabel('Noise variance')
plt.ylabel('Bit Error Rate')
plt.savefig('BER_sigma_code2.jpg')
plt.show()

BER = []
Eb_N0_dB = [-2, 0, 2, 4, 6]
Eb_N0 = [10**(i/10) for i in Eb_N0_dB] #corresponding values of E_b/N_0

print("For varying E_b/N_0 values....")

for t in range(len(Eb_N0_dB)):
	print("For Eb_N0 =", Eb_N0_dB[t], "dB")
	N_0 = E_b/Eb_N0[t]
	w = WGN(f_s*N_0/2, code_len) #white gaussian noise

	r = s + w #received signal

	#demodulating scheme	
	r_sym = np.zeros((int(code_len/2), 2))
	for i in range(len(r_sym)):
	    r_sym[i] = decomp(r[i]) #converting into signal space co-ordinates

	dist = np.zeros((len(r_sym), M)) #matrix to store the distance values from the 4 symbols
	for i in range(len(r)):
	    for j in range(M):
		    dist[i][j] = np.linalg.norm(r_sym[i] - S_space[j]) #calculating the distance
    
	demod = np.zeros((len(code_img)))
	i = 0
	while i < len(demod)/2:
	    index = np.argmin(dist[i]) #minimum distance 
	    demod[2*i], demod[2*i+1] = bit_array[index][0], bit_array[index][1]
	    i += 1

	decod = channel_decode_2(demod)
        
	error = (img != decod).sum()
	decod = decod.reshape(lx, ly)
	print("No. of incorrectly demodulated bits:", error)
	BER.append(error/size)
	print("Bit Error rate:", error/size, "\n")

	plt.imshow(decod, 'gray')
	plt.title('For E_b/N_0 = '+str(Eb_N0_dB[t])+'dB')
	plt.savefig('decoded_img_code2_EbN0='+str(Eb_N0_dB[t])+"dB.jpg")
	plt.show()

plt.semilogy(Eb_N0_dB, BER)
plt.xlabel('$E_b/N_0$ in dB')
plt.ylabel('Bit Error Rate')
plt.savefig('BER_EbN0_code2.jpg')
plt.show()




