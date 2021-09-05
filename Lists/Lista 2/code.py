from tabulate import tabulate
from scipy.sparse import csr_matrix
from numba import njit, prange

import math
import numpy as np

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
			
	return b

def sor(A, b, omega, x0, tol, norm = np.inf, max_iter = 10e4):
	'''
	solves Ax = b system with SOR method with w = omega
	and tolerance tol in the given norm 
	'''
	
	if norm == np.inf:
		norm = lambda x : np.linalg.norm(x, np.inf)
	elif type(norm) == int:
		norm = lambda x : np.linalg.norm(x, norm)
	
	x = x0.copy()
	error = norm(A @ x - b.T)
	n_iter = 0
	while error > tol and n_iter < max_iter:
		for i in range(A.shape[0]):
			mult = 0
			for j in range(A.shape[1]):
				if j != i:
					mult += A[i, j] * x[j]
			
			x[i] = (1 - omega) * x[i] + omega / A[i, i] * (b[i] - mult)
		
		error = norm(A @ x - b.T)
		n_iter += 1
		
	return x, error, n_iter

def expert_sor(A, b, omega, x0, tol, norm = np.inf, max_iter = 10e4):
	'''
	solves Ax = b system with SOR method with w = omega
	and tolerance tol in the given norm 
	'''
	
	if norm == np.inf:
		norm = lambda x : np.linalg.norm(x, np.inf)
	elif type(norm) == int:
		norm = lambda x : np.linalg.norm(x, norm)
	
	x = x0.copy()
	error = norm(A @ x - b.T)
	n_iter = 0
	dim = A.shape[0]
	m = int(math.sqrt(dim))
	while error > tol and n_iter < max_iter:
		for i in range(A.shape[0]):
			mult = 0
			if i - 1 >= 0 and i % m != 0:
				mult -= x[i - 1] # this A[i, j] is -1
			
			if i + 1 < dim and i % m != (m - 1):
				mult -= x[i + 1] # this A[i, j] is -1
			
			if i - m >= 0:
				mult -= x[i - m] # this A[i, j] is -1
			
			if i + m < dim:
				mult -= x[i + m] # this A[i, j] is -1
			
			'''
			for j in range(A.shape[1]):
				if j != i:
					mult += A[i, j] * x[j]
			'''
			
			x[i] = (1 - omega) * x[i] + omega / A[i, i] * (b[i] - mult)
		
		error = norm(A @ x - b.T)
		n_iter += 1
		
	return x, error, n_iter



# finding optimal omega
omega_opt = lambda m : 2 / (1 + math.sin(np.pi / (m + 1)))
def tests(ms, omegas, tol, mean, sd, max_iter):
	n_iters = [[ms[i]] + list(np.zeros(len(omegas), int)) for i in range(len(ms))]
	errors = [[ms[i]] + list(np.zeros(len(omegas), int)) for i in range(len(ms))]
	solutions = [[ms[i]] + list(np.zeros(len(omegas), int)) for i in range(len(ms))]
	count = 1
	for i in range(len(ms)):
		m = ms[i]
		A = create_A(m)
		b = create_b(m)
		x0 = np.random.normal(mean, sd, A.shape[0])
		for j in range(len(omegas)):
			omega = omegas[j]
			if omega == omega_opt:
				omega = omega_opt(m)
				
			print(count, m, omega)
			x0_j = x0.copy()
			x, error, n_iter = expert_sor(A, b, omega, x0_j, tol, np.inf, max_iter = max_iter)
			n_iters[i][j + 1] = n_iter
			errors[i][j + 1] = error
			solutions[i][j + 1] = x
			count += 1
	
	header = ['m / ω', 'ω ótimo'] + omegas[1:]
	print('\nIterations:')
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
	print()

mean = 2
sd = 0.01
tol = 10e-7 # 0.000 001
ms = [10, 50, 1000, 5000]
omegas = [omega_opt, 1, .5]
max_iter = 10e8 # 1 000 000 000
tests(ms, omegas, tol, mean, sd, max_iter)
