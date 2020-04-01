# !/usr/bin/env python3

#Implementation of rate-1/4 regular LDPC code for a BSC

import numpy as np 
import matplotlib.pyplot as plt 
import random

img = np.load('binary_image.npy')
lx, ly = len(img), len(img[0])
img = np.array(img).flatten()
size = lx*ly
n, k = 20, 5
m = n-k

def swap_columns(a,b,arrayIn):
  """
  Swaps two columns in a matrix.
  """
  arrayOut = arrayIn.copy()
  arrayOut[:,a] = arrayIn[:,b]
  arrayOut[:,b] = arrayIn[:,a]
  return arrayOut

def move_row_to_bottom(i,arrayIn):
  """"
  Moves a specified row (just one) to the bottom of the matrix,
  then rotates the rows at the bottom up.
  """
  arrayOut = arrayIn.copy()
  numRows = arrayOut.shape[0]
  # Push the specified row to the bottom.
  arrayOut[numRows-1] = arrayIn[i,:]
  # Now rotate the bottom rows up.
  index = 2
  while (numRows-index) >= i:
    arrayOut[numRows-index,:] = arrayIn[numRows-index+1]
    index = index + 1
  return arrayOut

def generator(H):
  """
  This function finds the systematic form of the generator
  matrix H. This form is G = [I P] where I is an identity
  matrix and P is the parity submatrix. If the H matrix
  provided is not full rank, then dependent rows will be deleted.
  This function does not convert parity check (H) matrices to the
  generator matrix format. Use the function generatorFromH
  for that purpose.
  """
  tempArray = H.copy()
  numRows = tempArray.shape[0]
  numColumns = tempArray.shape[1]
  limit = numRows
  rank = 0
  i = 0
  while i < limit:
    # Flag indicating that the row contains a non-zero entry
    found = False
    for j in np.arange(i, numColumns):
      if tempArray[i, j] == 1:
        # Encountered a non-zero entry at (i, j)
        found = True
        # Increment rank by 1
        rank = rank + 1
        # make the entry at (i,i) be 1
        tempArray = swap_columns(j,i,tempArray)
        break
    if found == True:
      for k in np.arange(0,numRows):
        if k == i: continue
        # Checking for 1's
        if tempArray[k, i] == 1:
          # add row i to row k
          tempArray[k,:] = tempArray[k,:] + tempArray[i,:]
          # Addition is mod2
          tempArray = tempArray.copy() % 2
          # All the entries above & below (i, i) are now 0
      i = i + 1
    if found == False:
      # push the row of 0s to the bottom, and move the bottom
      # rows up (sort of a rotation thing)
      tempArray = move_row_to_bottom(i,tempArray)
      # decrease limit since we just found a row of 0s
      limit -= 1
      # the rows below i are the dependent rows, which we discard
  G = tempArray[0:i,:]
  return G

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

    count = 0
    for i in range(0, l2):
        for j in range(wr*i, wr*(i+1)):
            H[i][j] = 1

    for i in range(l2, m):
        if(i%l2 == 0):
            index_arr = permut_index(n)
        H[i, range(n)] = H[i%l2, index_arr]

    return H

def encode(msg, G): #function for encoding
  code = np.zeros((int(len(msg)*n/k)))
  i, j = 0, 0
  while i < len(msg):
      code[j:j+n] = (np.matmul(G.T, msg[i:i+k]))%2
      i += k
      j += n
  return code


H = pc_matrix()
G = generator(H)
code_img = encode(img, G)
print(np.matmul(H[:, 0:n-2], G.T))