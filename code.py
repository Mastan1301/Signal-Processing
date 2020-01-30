import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import norm

#0 represents a black pixel and 1 is a white pixel.
T = 1e-6
f_s, f_c = 50e6, 2e6
n = int(T * f_s)

#The basis functions in the signal space representation.

img = np.load('binary_image.npy')
lx, ly = len(img), len(img[0])
plt.imshow(img, 'gray')
plt.savefig('original_img.jpg')
plt.show()

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

def WGN(N_0):
    s1, s2 = int(size/2), int(T*f_s)
    mu, sigma = 0, np.sqrt(f_s*N_0/2)
    res = np.zeros((s1, s2))
    for i in range(s1):
        res[i] = np.random.normal(mu, sigma, s2)
    return res

'''def E(x):
    return (np.linalg.norm(x))**2'''

def decomp(x):
    cos = [np.cos(2*np.pi*f_c*i)*np.sqrt(2/T) for i in np.linspace(0, T, n)]
    sin = [np.sin(2*np.pi*f_c*i)*np.sqrt(2/T) for i in np.linspace(0, T, n)]
    res = [np.dot(x, cos), np.dot(x, sin)]
    return res

s = np.zeros(((int(size/2)), int(T*f_s)))
for i in range(int(size/2)):
    s[i] = mod(img[2*i], img[2*i+1], i) #modulating

#Here S_space entries are signal-space co-ordinates.
S_space = np.array([[-1, 1], [-1, -1], [1, -1], [1, 1]])
bit_array = np.array([[1, 0], [1, 1], [0, 1], [0, 0]])
M = len(S_space)

#Manually computed average energy using integration
E_avg = T

#Computing average energy per bit
E_b = E_avg/np.log2(M)
Eb_N0_dB = [-10, -5, 0, 5] #values of E_b/N_0 in dB
Eb_N0 = [10**(i/10) for i in Eb_N0_dB] #corresponding values of E_b/N_0
BER = []

for k in range(len(Eb_N0)):
	print("For E_b/N_0 =", Eb_N0_dB[k], "dB")
	N_0 = E_b/Eb_N0[k]
	w = WGN(N_0) #white gaussian noise

	#demodulating scheme
	r = s + w #received signal
	r_sym = np.zeros((int(size/2), 2))
	for i in range(len(r_sym)):
	    r_sym[i] = decomp(r[i]) #converting into signal space co-ordinates

	dist = np.zeros((len(r_sym), M)) #matrix to store the distance values from the 4 symbols
	for i in range(len(r)):
	    for j in range(M):
		    dist[i][j] = np.linalg.norm(r_sym[i] - S_space[j]) #calculating the distance
    
	demod = np.zeros((size))
	i = 0
	while i < len(demod)/2:
	    index = np.argmin(dist[i]) #minimum distance 
	    demod[2*i], demod[2*i+1] = bit_array[index][0], bit_array[index][1]
	    i += 1

	#Verification
	error = (img != demod).sum()
	print("No. of incorrectly demodulated bits:", error)
	BER.append(error/size)
	print("Bit Error rate:", error/size)

	Q = 1 - norm.cdf(np.sqrt(2*Eb_N0[k])) #value of Q(sqrt(2*E_b/N_0))
	print("Q-function:", Q, "\n")
	demod = demod.reshape(lx, ly)
	plt.imshow(demod, 'gray')
	plt.savefig('demod_img_EbN0='+str(Eb_N0_dB[k])+"dB.jpg")
	plt.show()

plt.stem(range(len(Eb_N0)), BER, use_line_collection=True)
plt.xlabel('$E_b/N_0$ in dB')
plt.ylabel('Bit Error Rate')
plt.savefig('BER.jpg')
plt.show()


