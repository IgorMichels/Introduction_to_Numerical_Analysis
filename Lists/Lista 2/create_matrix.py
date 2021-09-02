import numpy as np
from numba import njit, prange
from scipy.sparse import csr_matrix
from time import time

def create_matrix(m):
	'''
	creates matrix A in R^(m² x m²) such that
		⌈  T  -I        			  ⌉
   		|-I   T  -I				  |
		|    -I   T  -I			  |
		|						  |
	A = |		    dots		  |
		|		       dots       |
		|						  |
		|					 	-I|
		⌊ 					-I   T⌋ 
		
	with I equal identity in R^(m x m) and T in R^(m x m) such that
		⌈  4  -1        			  ⌉
   		|-1   4  -1				  |
		|    -1   4  -1			  |
		|						  |
	T = |		    dots		  |
		|		       dots       |
		|						  |
		|					 	-1|
		⌊ 					-1   4⌋ 
	'''
	
	dim = m**2
	row = [0 for i in range(5 * dim - 2)]
	col = [0 for i in range(5 * dim - 2)]
	val = [0 for i in range(5 * dim - 2)]
	counter = 0
	for i in range(dim):
		row[counter] = i
		col[counter] = i
		val[counter] = 4
		counter += 1
		if i - 1 >= 0:
			row[counter] = i
			col[counter] = i - 1
			val[counter] = -1
			counter += 1
		
		if i + 1 < dim:
			row[counter] = i
			col[counter] = i + 1
			val[counter] = -1
			counter += 1
		
		if i - m >= 0:
			row[counter] = i
			col[counter] = i - m
			val[counter] = -1
			counter += 1
		
		if i + m < dim:
			row[counter] = i
			col[counter] = i + m
			val[counter] = -1
			counter += 1
			
	return csr_matrix((val, (row, col)), shape = (dim, dim))
