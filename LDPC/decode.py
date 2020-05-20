import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import norm

T = 1e-6
f_s, f_c = 50e6, 2e6
n1 = int(T * f_s)
img = np.load('./binary_image.npy.npy')
lx, ly = len(img), len(img[0])

code = np.loadtxt("./encoded_bits.dat")
code_len = len(code)
G, H = np.loadtxt("./G.dat"), np.loadtxt("./H.dat")

k, n = len(G), 1
M = 4

img = np.array(img).flatten() #vectorising the matrix
size = len(img)

def bit(b):
    if b == 1:
        return -1
    return 1

def mod(b1, b2, i): # modulation function
    t = np.linspace((i-1)*T, i*T, int(T*f_s))
    res = np.zeros(len(t))
    x1 = bit(b1)
    x2 = bit(b2)
    for j in range(len(t)):
        res[j] = (x1 * np.cos(2*np.pi*f_c*t[j])) + (x2 * np.sin(2*np.pi*f_c*t[j]))
    return res

def WGN(variance, length): # function to generate white-gaussian-noise
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

def belief_prop(demod): # decoding using belief-propagation/sum-product algorithm


s = np.zeros(((int(code_len/2)), int(T*f_s)))
for i in range(int(code_len/2)):
    s[i] = mod(code[2*i], code[2*i+1], i) #modulating

#Here S_space entries are signal-space co-ordinates.
S_space = np.array([[-1, 1], [-1, -1], [1, -1], [1, 1]])
bit_array = np.array([[1, 0], [1, 1], [0, 1], [0, 0]])
M = len(S_space)

#Manually computed average energy using integration
E_avg = T

#Computing average energy per bit
E_b =(E_avg*n)/(np.log2(M)*k)
N_0 = 1e6
w = WGN(f_s*N_0/2, code_len)

r = s + w #received signal

#demodulating scheme	
r_sym = np.zeros((int(code_len/2), 2))
for i in range(len(r_sym)):
    r_sym[i] = decomp(r[i]) #converting into signal space co-ordinates

dist = np.zeros((len(r_sym), M)) #matrix to store the distance values from the 4 symbols
for i in range(len(r)):
    for j in range(M):
        dist[i][j] = np.linalg.norm(r_sym[i] - S_space[j]) #calculating the distance

demod = np.zeros((code_len))
i = 0
while i < len(demod)/2:
    index = np.argmin(dist[i]) #minimum distance demodulation
    demod[2*i], demod[2*i+1] = bit_array[index][0], bit_array[index][1]
    i += 1

decod = belief_prop(demod) #decoding 
    
error = (img != decod).sum()
decod = decod.reshape(lx, ly)
print("No. of incorrectly demodulated bits:", error)
print("Bit Error rate:", error/size, "\n")


