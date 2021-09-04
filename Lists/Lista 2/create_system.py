import numpy as np
from scipy.sparse import csr_matrix
from numba import njit, prange

def create_A(m):
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
	row = np.zeros(5 * dim - 2)
	col = np.zeros(5 * dim - 2)
	val = np.zeros(5 * dim - 2)
	counter = 0
	for i in range(dim):
		row[counter] = i
		col[counter] = i
		val[counter] = 4
		counter += 1
		if i - 1 >= 0 and i % m != 0:
			row[counter] = i
			col[counter] = i - 1
			val[counter] = -1
			counter += 1
		
		if i + 1 < dim and i % m != (m - 1):
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

@njit(fastmath = True, cache = True, parallel = True)
def create_b(m):
	'''
	creates vector b such that Ax = b, where A is like above
	and x = [2, 2, ..., 2]
	'''
	dim = m**2
	b = 8 * np.ones(dim)
	for i in prange(dim):
		if i - 1 >= 0 and i % m != 0:
			b[i] -= 2
		
		if i + 1 < dim and i % m != (m - 1):
			b[i] -= 2
		
		if i - m >= 0:
			b[i] -= 2
		
		if i + m < dim:
			b[i] -= 2
			
	return b
