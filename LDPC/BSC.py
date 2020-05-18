# !/usr/bin/env python3

#Implementation of rate-1/4 (approximately) regular LDPC code for a BSC

import numpy as np 
import matplotlib.pyplot as plt 
import random
from numpy.random import shuffle, randint
from numpy.linalg import inv, det

img = np.load('binary_image.npy')
lx, ly = len(img), len(img[0])
img = np.array(img).flatten()
size = lx*ly
n, k = 8, 3
p, q = 3, 2 #weights of columns and rows respectively
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

def generator(H, verbose=False):
  """
  If given a parity check matrix H, this function returns a
  gaussian_elimination matrix G in the systematic form: G = [I P]
    where:  I is an np.identity matrix, size k x k
            P is the parity submatrix, size k x (n-k)
  If the H matrix provided is not full rank, then dependent rows
  will be deleted first.
  """
  if verbose:
    print('received H with size: ', H.shape)

  # First, put the H matrix into the form H = [I|m] where:
  #   I is (n-k) x (n-k) np.identity matrix
  #   m is (n-k) x k
  # This part is just copying the algorithm from getSystematicGmatrix
  tempArray = gaussian_elimination(H)
  #tempArray = H
  # Next, swap I and m columns so the matrix takes the forms [m|I].
  n      = H.shape[1]
  k      = n - H.shape[0]
  I_temp = tempArray[:,0:(n-k)]
  m      = tempArray[:,(n-k):n]
  newH   = np.concatenate((m,I_temp),axis=1)

  # Now the submatrix m is the transpose of the parity submatrix,
  # i.e. H is in the form H = [P'|I]. So G is just [I|P]
  k = m.shape[1]
  G = np.concatenate((np.identity(k),m.T),axis=1)
  if verbose:
    print('returning G with size: ', G.shape)
  return G

def gaussian_elimination(H):
  """
  This function finds the systematic form of the gaussian_elimination
  matrix H. This form is G = [I P] where I is an np.identity
  matrix and P is the parity submatrix. If the H matrix
  provided is not full rank, then dependent rows will be deleted.
  This function does not convert parity check (H) matrices to the
  gaussian_elimination matrix format. Use the function generatorFromH
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

def permut_index(s): #returns an np.array of randomly shuffled indices
    arr = list(range(s))
    res = []
    while len(arr):
        temp = random.choice(arr)
        res.append(temp)
        arr.remove(temp)
    return res

def pc_matrix(): #parity check matrix    
    """
    This function constructs a LDPC parity check matrix
    H. The algorithm follows Gallager's approach where we create
    p submatrices and stack them together. Reference: Turbo
    Coding for Satellite and Wireless Communications, section
    9,3.
    Note: the matrices computed from this algorithm will never
    have full rank. (Reference Gallager's Dissertation.) They
    will have rank = (number of rows - p + 1). To convert it
    to full rank, use the function get_full_rank_H_matrix
    """
    # TODO: There should probably be other guidelines for n/p/q,
    # but I have not found any specifics in the literature....

    # For this algorithm, n/p must be an integer, because the
    # number of rows in each submatrix must be a whole number.
    ratioTest = (n*1.0) / q
    if ratioTest%1 != 0:
      print('\nError in regular_LDPC_code_contructor: The ', end='')
      print('ratio of inputs n/q must be a whole number.\n')
      return

    # First submatrix first:
    m = ((n*p) / q)+(p-1)  # number of rows in H matrix
    submatrix1 = np.zeros((int(m / p),n))
    for row in np.arange(int(m / p)):
      range1 = row*q
      range2 = (row+1)*q
      submatrix1[row,range1:range2] = 1
      H = submatrix1

    # Create the other submatrices and vertically stack them on.
    submatrixNum = 2
    newColumnOrder = np.arange(n)
    while submatrixNum <= p:
      submatrix = np.zeros((int(m / p),n))
      shuffle(newColumnOrder)

      for columnNum in np.arange(n):
        submatrix[:,columnNum] = \
                                 submatrix1[:,newColumnOrder[columnNum]]

      H = np.vstack((H,submatrix))
      submatrixNum = submatrixNum + 1

    # Double check the row weight and column weights.
    size = H.shape
    rows = size[0]
    cols = size[1]

    # Check the row weights.
    for rowNum in np.arange(rows):
      nonzeros = np.array(H[rowNum,:].nonzero())
      if nonzeros.shape[1] != q:
        print('Row', rowNum, 'has incorrect weight!')
        return

    # Check the column weights
    for columnNum in np.arange(cols):
      nonzeros = np.array(H[:,columnNum].nonzero())
      if nonzeros.shape[1] != p:
        print('Row', columnNum, 'has incorrect weight!')
        return

    return H

def encode(msg, G): #function for encoding
  k = G.shape[0]
  code = np.zeros((int(len(msg)*n/k)))
  i, j = 0, 0
  while i+k < len(msg):
      code[j:j+n] = (np.matmul(G.T, msg[i:i+k]))%2
      i += k
      j += n

  return code

H = gaussian_elimination(pc_matrix())
G = generator(H)
code_img = encode(img, G)
np.savetxt("encoded_bits.dat", code_img)
#print(np.dot(G, H.T)%2)