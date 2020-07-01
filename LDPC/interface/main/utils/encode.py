# !/usr/bin/env python3

#Implementation of regular LDPC code for a BSC

from numpy import *
import matplotlib.pyplot as plt 
import random
from numpy.random import shuffle, randint
from numpy.linalg import inv, det

n = 20
p, q = 4, 5 #weights of columns (w_c) and rows(w_r) respectively

def pc_matrix():
    ratioTest = (n*1.0) / q
    if ratioTest%1 != 0:
      return

    # First submatrix first:
    m = (n*p) / q  # number of rows in H matrix
    submatrix1 = zeros((int(m / p),n))
    for row in arange(int(m / p)):
      range1 = row*q
      range2 = (row+1)*q
      submatrix1[row,range1:range2] = 1
      H = submatrix1

    # Create the other submatrices and vertically stack them on.
    submatrixNum = 2
    newColumnOrder = arange(n)
    while submatrixNum <= p:
      submatrix = zeros((int(m / p),n))
      shuffle(newColumnOrder)

      for columnNum in arange(n):
        submatrix[:,columnNum] = \
                                 submatrix1[:,newColumnOrder[columnNum]]

      H = vstack((H,submatrix))
      submatrixNum = submatrixNum + 1

    # Double check the row weight and column weights.
    size = H.shape
    rows = size[0]
    cols = size[1]

    # Check the row weights.
    for rowNum in arange(rows):
      nonzeros = array(H[rowNum,:].nonzero())
      if nonzeros.shape[1] != q:
        return

    # Check the column weights
    for columnNum in arange(cols):
      nonzeros = array(H[:,columnNum].nonzero())
      if nonzeros.shape[1] != p:
        return

    return H

def pad_zeros(arr, k):
  i = [0 for _ in range(k)]
  return np.concatenate(arr, np.array(i))
    

def swap_columns(a,b,arrayIn):
  """
  Swaps two columns in a matrix.
  """
  arrayOut = arrayIn.copy()
  arrayOut[:,a] = arrayIn[:,b]
  arrayOut[:,b] = arrayIn[:,a]
  return arrayOut

def move_row_to_bottom(i,arrayIn):
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

def getSystematicGmatrix(GenMatrix):
  
  tempArray = GenMatrix.copy()
  numRows = tempArray.shape[0]
  numColumns = tempArray.shape[1]
  limit = numRows
  rank = 0
  i = 0
  while i < limit:
    # Flag indicating that the row contains a non-zero entry
    found = False
    for j in arange(i, numColumns):
      if tempArray[i, j] == 1:
        # Encountered a non-zero entry at (i, j)
        found = True
        # Increment rank by 1
        rank = rank + 1
        # make the entry at (i,i) be 1
        tempArray = swap_columns(j,i,tempArray)
        break
    if found == True:
      for k in arange(0,numRows):
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

def generator(H, verbose=False):
  # First, put the H matrix into the form H = [I|m] where:
  #   I is (n-k) x (n-k) identity matrix
  #   m is (n-k) x k
  # This part is just copying the algorithm from getSystematicGmatrix
  tempArray = getSystematicGmatrix(H)

  # Next, swap I and m columns so the matrix takes the forms [m|I].
  n      = tempArray.shape[1]
  k      = n - tempArray.shape[0]
  I_temp = tempArray[:,0:(n-k)]
  m      = tempArray[:,(n-k):n]

  newH   = concatenate((m,I_temp),axis=1)

  # Now the submatrix m is the transpose of the parity submatrix,
  # i.e. H is in the form H = [P'|I]. So G is just [I|P]
  G = concatenate((identity(k),m.T),axis=1)
  return G

def get_full_rank_H_matrix(H, verbose=False):
  tempArray = H.copy()
  if linalg.matrix_rank(tempArray) == tempArray.shape[0]:
    if verbose:
      print('Returning H; it is already full rank.')
    return tempArray

  numRows = tempArray.shape[0]
  numColumns = tempArray.shape[1]
  limit = numRows
  rank = 0
  i = 0

  # Create an array to save the column permutations.
  columnOrder = arange(numColumns).reshape(1,numColumns)

  # Create an array to save the row permutations. We just need
  # this to know which dependent rows to delete.
  rowOrder = arange(numRows).reshape(numRows,1)

  while i < limit:
    if verbose:
      print('In get_full_rank_H_matrix; i:', i)
      # Flag indicating that the row contains a non-zero entry
    found  = False
    for j in arange(i, numColumns):
      if tempArray[i, j] == 1:
        # Encountered a non-zero entry at (i, j)
        found =  True
        # Increment rank by 1
        rank = rank + 1
        # Make the entry at (i,i) be 1
        tempArray = swap_columns(j,i,tempArray)
        # Keep track of the column swapping
        columnOrder = swap_columns(j,i,columnOrder)
        break
    if found == True:
      for k in arange(0,numRows):
        if k == i: continue
        # Checking for 1's
        if tempArray[k, i] == 1:
          # Add row i to row k
          tempArray[k,:] = tempArray[k,:] + tempArray[i,:]
          # Addition is mod2
          tempArray = tempArray.copy() % 2
          # All the entries above & below (i, i) are now 0
      i = i + 1
    if found == False:
      # Push the row of 0s to the bottom, and move the bottom
      # rows up (sort of a rotation thing).
      tempArray = move_row_to_bottom(i,tempArray)
      # Decrease limit since we just found a row of 0s
      limit -= 1
      # Keep track of row swapping
      rowOrder = move_row_to_bottom(i,rowOrder)

  # Don't need the dependent rows
  finalRowOrder = rowOrder[0:i]

  # Reorder H, per the permutations taken above .
  # First, put rows in order, omitting the dependent rows.
  newNumberOfRowsForH = finalRowOrder.shape[0]
  newH = zeros((newNumberOfRowsForH, numColumns))
  for index in arange(newNumberOfRowsForH):
    newH[index,:] = H[finalRowOrder[index],:]

  # Next, put the columns in order.
  tempHarray = newH.copy()
  for index in arange(numColumns):
    newH[:,index] = tempHarray[:,columnOrder[0,index]]

  return newH

def encode(msg, G): #function for encoding
  k = G.shape[0]
  code = zeros((int(len(msg)*n/k)))
  i, j = 0, 0
  while i+k < len(msg):
      code[j:j+n] = (matmul(G.T, msg[i:i+k]))%2
      i += k
      j += n

  return code

def get_full_rank_H_matrix(H, verbose=False):
  tempArray = H.copy()
  if linalg.matrix_rank(tempArray) == tempArray.shape[0]:
    return tempArray

  numRows = tempArray.shape[0]
  numColumns = tempArray.shape[1]
  limit = numRows
  rank = 0
  i = 0

  # Create an array to save the column permutations.
  columnOrder = arange(numColumns).reshape(1,numColumns)

  # Create an array to save the row permutations. We just need
  # this to know which dependent rows to delete.
  rowOrder = arange(numRows).reshape(numRows,1)

  while i < limit:
      # Flag indicating that the row contains a non-zero entry
    found  = False
    for j in arange(i, numColumns):
      if tempArray[i, j] == 1:
        # Encountered a non-zero entry at (i, j)
        found =  True
        # Increment rank by 1
        rank = rank + 1
        # Make the entry at (i,i) be 1
        tempArray = swap_columns(j,i,tempArray)
        # Keep track of the column swapping
        columnOrder = swap_columns(j,i,columnOrder)
        break
    if found == True:
      for k in arange(0,numRows):
        if k == i: continue
        # Checking for 1's
        if tempArray[k, i] == 1:
          # Add row i to row k
          tempArray[k,:] = tempArray[k,:] + tempArray[i,:]
          # Addition is mod2
          tempArray = tempArray.copy() % 2
          # All the entries above & below (i, i) are now 0
      i = i + 1
    if found == False:
      # Push the row of 0s to the bottom, and move the bottom
      # rows up (sort of a rotation thing).
      tempArray = move_row_to_bottom(i,tempArray)
      # Decrease limit since we just found a row of 0s
      limit -= 1
      # Keep track of row swapping
      rowOrder = move_row_to_bottom(i,rowOrder)

  # Don't need the dependent rows
  finalRowOrder = rowOrder[0:i]

  # Reorder H, per the permutations taken above .
  # First, put rows in order, omitting the dependent rows.
  newNumberOfRowsForH = finalRowOrder.shape[0]
  newH = zeros((newNumberOfRowsForH, numColumns))
  for index in arange(newNumberOfRowsForH):
    newH[index,:] = H[finalRowOrder[index],:]

  # Next, put the columns in order.
  tempHarray = newH.copy()
  for index in arange(numColumns):
    newH[:,index] = tempHarray[:,columnOrder[0,index]]

  return newH

def main(img):
  lx, ly = len(img), len(img[0])
  img = array(img).flatten()
  size = lx*ly
  H = pc_matrix()
  G = generator(H)
  code_img = encode(img, G)
  savetxt("./media/data/encoded_bits.dat", code_img)
  savetxt("./media/data/G.dat", G)
  savetxt("./media/data/H.dat", H)
