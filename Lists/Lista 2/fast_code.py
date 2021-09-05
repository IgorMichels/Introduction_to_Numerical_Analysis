from tabulate import tabulate
from scipy.sparse import csr_matrix
from numba import njit, prange

import math
import numpy as np

@njit(fastmath = True, cache = True, parallel = False)
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
	row = []
	col = []
	for i in range(dim):
		if i - 1 >= 0 and i % m != 0:
			row.append(i)
			col.append(i - 1)
		
		if i + 1 < dim and i % m != (m - 1):
			row.append(i)
			col.append(i + 1)
		
		if i - m >= 0:
			row.append(i)
			col.append(i - m)
		
		if i + m < dim:
			row.append(i)
			col.append(i + m)
			
	return (np.array(row), np.array(col))

@njit(fastmath = True, cache = True, parallel = True)
def create_b(m):
	'''
	creates vector b such that Ax = b, where A is like above
	and x = [2, 2, ..., 2]
	'''
	dim = m**2
	b = 8 * np.ones((dim, 1))
	for i in prange(dim):
		if i - 1 >= 0 and i % m != 0:
			b[i] -= 2
		
		if i + 1 < dim and i % m != (m - 1):
			b[i] -= 2
		
		if i - m >= 0:
			b[i] -= 2
		
		if i + m < dim:
			b[i] -= 2
			
	return b.T

@njit(fastmath = True, cache = True, parallel = True)
def mult_mat(A, x):
	row, col = A
	result = np.zeros(x.shape[0])
	for i in prange(len(row)):
		result[row[i]] -= x[col[i]]
		
	for i in prange(x.shape[0]):
		result[i] += 4 * x[i]
	
	return result
	
@njit(fastmath = True, cache = True)
def norm(x):
	return np.abs(x).max()
	
@njit(fastmath = True, cache = True, parallel = False)
def smart_sor(A, b, omega, x0, tol, max_iter = 1e5):
	row, col = A
	x = x0.copy()
	result = mult_mat(A, x)
	error = norm(result - b)
	n_iter = 0
	dim = b.shape[1]
	m = int(math.sqrt(dim))
	while error > tol and n_iter < max_iter:
		for i in range(dim):
			mult = 0
			if i - 1 >= 0 and i % m != 0:
				mult -= x[i - 1] # this A[i, j] is -1
			
			if i + 1 < dim and i % m != (m - 1):
				mult -= x[i + 1] # this A[i, j] is -1
			
			if i - m >= 0:
				mult -= x[i - m] # this A[i, j] is -1
			
			if i + m < dim:
				mult -= x[i + m] # this A[i, j] is -1
			
			x[i] = (1 - omega) * x[i] + omega / 4 * (b[0, i] - mult) # A[i, i] = 4 for all i
		
		result = mult_mat(A, x)
		error = norm(result - b)
		n_iter += 1
		if n_iter % 1000 == 0:
			print(error)
		
	return x, error, n_iter

# finding optimal omega
@njit(fastmath = True, cache = True)
def omega_opt(m):
	return 2 / (1 + math.sin(np.pi / (m + 1)))

#@njit(fastmath = True, cache = True, parallel = True)
def tests(ms, omegas, tol, mean, max_iter = 1e5):
	n_iters = []
	errors = []
	solutions = []
	x0s = []
	count = 1
	for i in range(len(ms)):
		m = ms[i]
		n_iters.append([m])
		errors.append([m])
		solutions.append([m])
		x0s.append([m])
		A = create_A(m)
		b = create_b(m)
		sd = 1/(20 * m)
		x0 = np.random.normal(mean, sd, m**2)
		for j in range(len(omegas)):
			omega = omegas[j]
			if omega == 100:
				omega_in_use = omega_opt(m)
			else:
				omega_in_use = omega
				
			print(count, m, omega_in_use)
			x0_j = x0.copy()
			x, error, n_iter = smart_sor(A, b, omega_in_use, x0_j, tol, max_iter)
			n_iters[i].append(n_iter)
			errors[i].append(error)
			solutions[i].append(x)
			x0s[i].append(x0_j)
			count += 1
			
		print(n_iters[i])
		
	header = ['m / ω', 'ω ótimo'] + omegas[1:]
	print('\n\nIterations:')
	print(tabulate(n_iters,
				   headers = header,
				   tablefmt = 'fancy_grid'))

	print('\nLaTeX code:')
	print(tabulate(n_iters,
				   headers = header,
				   tablefmt = 'latex'))

	print('\nErrors:')
	print(tabulate(errors,
				   headers = header,
				   tablefmt = 'fancy_grid'))

	print('\nLaTeX code:')
	print(tabulate(errors,
				   headers = header,
				   tablefmt = 'latex'))
	
	return n_iters, errors, solutions, x0s

mean = 2
tol = 1e-6 # 0.000 001
ms = [5 * i for i in range(1, 11)]
omegas = [100, 1, .5]
n_iters, errors, solutions, x0s = tests(ms, omegas, tol, mean)

mean = 2
tol = 1e-6 # 0.000 001
ms = [10, 50, 1000, 5000]
omegas = [100, 1, .5]
n_iters, errors, solutions, x0s = tests(ms, omegas, tol, mean)

