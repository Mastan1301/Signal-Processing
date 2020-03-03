#EE18BTECH11039, EE18BTECH11010

#Implementation of rate-1/2 linear channel coding and decoding for varying values of noise variance

import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import norm

T = 1e-6
f_s, f_c = 50e6, 2e6
n1 = int(T * f_s)
n, k = 8, 4

img = np.load('mss.npy')
lx, ly = len(img), len(img[0])

img = np.array(img).flatten() #vectorising the matrix
size = len(img)
g1 = np.array([[1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 1, 1, 0, 0], [1, 0, 1, 0, 1, 0, 1, 0], [0, 1, 1, 0, 1, 0, 0, 1]])

def bit(b):
    if b == 1:
        return -1
    return 1

def pad(arr, k): #for padding the zeros
    arr = arr[::-1]
    for _ in range(k-len(arr)):
        arr = arr+'0'
    arr = arr[::-1]
    return arr

def gen_all_bits(arr, k): #generate all possible binary strings of length = k
    for i in range(int(2**k)):
        temp = pad(np.binary_repr(i), k)
        arr[i] = [int(j) for j in temp]

def mod(b1, b2, i):
    t = np.linspace((i-1)*T, i*T, int(T*f_s))
    res = np.zeros(len(t))
    x1 = bit(b1)
    x2 = bit(b2)
    for j in range(len(t)):
        res[j] = (x1 * np.cos(2*np.pi*f_c*t[j])) + (x2 * np.sin(2*np.pi*f_c*t[j]))
    return res

def WGN(variance, length):
    s1, s2 = int(length/2), int(T*f_s)
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

def channel_code_1(m, n, k):
    code = np.zeros((int(len(m)*n/k)))
    i, j = 0, 0
    while i < len(m):
        code[j:j+n] = (np.matmul(g1.T, m[i:i+k]))%2
        i += k
        j += n
    return code

all_bits = np.zeros((int(2**k), k))
gen_all_bits(all_bits, k)

def channel_decode_1(y, n, k):
    decode = np.zeros((int(size/k), k))
    ham_dist = np.zeros((int(len(y)/8), int(2**k)))
    i = 0
    while i < len(y):
        for j in range(len(all_bits)):
            temp = (np.matmul(g1.T, all_bits[j]))%2
            ham_dist[int(i/8)][j] = (temp != y[i:i+8]).sum()
        i += 8
    
    for i in range(len(decode)):
        decode[i] = all_bits[np.argmin(ham_dist[i])]

    decode = decode.flatten()
    return decode

code_img = channel_code_1(img, n, k) #channel coding
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

	decod = channel_decode_1(demod, n, k) #decoding 
        
	error = (img != decod).sum()
	decod = decod.reshape(lx, ly)
	print("No. of incorrectly demodulated bits:", error)
	BER.append(error/size)
	print("Bit Error rate:", error/size, "\n")

plt.semilogy(var, BER)
plt.xlabel('Noise Variance')
plt.ylabel('Bit Error Rate')
plt.savefig('BER_code1_sigma.jpg')
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

	decod = channel_decode_1(demod, n, k) #decoding
        
	error = (img != decod).sum()
	decod = decod.reshape(lx, ly)
	print("No. of incorrectly demodulated bits:", error)
	BER.append(error/size)
	print("Bit Error rate:", error/size, "\n")

	plt.imshow(decod, 'gray')
	plt.title('For E_b/N_0 = '+str(Eb_N0_dB[t])+'dB')
	plt.savefig('decoded_img_code1_EbN0='+str(Eb_N0_dB[t])+"dB.jpg")
	plt.show()

plt.semilogy(Eb_N0_dB, BER)
plt.xlabel('$E_b/N_0$ in dB')
plt.ylabel('Bit Error Rate')
plt.savefig('BER_EbN0_code1.jpg')
plt.show()









