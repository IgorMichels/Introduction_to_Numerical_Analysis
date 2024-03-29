import math
import numpy as np
import matplotlib.pyplot as plt
from time import time
from tabulate import tabulate
from numba import njit, prange, jit

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
	
def create_x0(m):
	'''
	creates vector x0 to SOR method
	'''
	dim = m**2
	x0 = 2 * np.ones(dim)
	if m <= 100:
		div = 10
	elif m <= 1000:
		div = 100
	else:
		div = 1000
	
	for i in range(0, dim, div):
		x0[i] = 1
	
	return x0

@njit(fastmath = True, cache = True, parallel = True)
def mult_mat(A, x):
	row, col = A
	# result of diag(A) * x
	result = 4 * x.copy() # diagonal
	for i in prange(len(row)):
		# update result with values from (A - diag(A)) * x
		result[row[i]] -= x[col[i]] # others values
	
	return result
	
@njit(fastmath = True, cache = True)
def norm(x):
	# only for compile function and run fast
	return np.abs(x).max()
	
@njit(fastmath = True, cache = True, parallel = False)
def smart_sor(A, b, omega, x0, tol, max_iter = 1e5):
	# set arrays to variables
	row, col = A
	
	# copy x0 to update x
	x = x0.copy()
	
	# initial error
	result = mult_mat(A, x)
	error = norm(result - b)
	
	# iterations count
	n_iter = 0
	
	# matrix dimension
	dim = b.shape[1]
	m = int(math.sqrt(dim))
	
	# create graph data
	ite = []
	dif = []
	
	# loop to apply SOR method
	while error > tol and n_iter < max_iter:
		for i in range(dim):
			# auxiliary variable to update x[i]
			mult = 0
			if i - 1 >= 0 and i % m != 0:
				mult -= x[i - 1] # this A[i, j] is -1
			
			if i + 1 < dim and i % m != (m - 1):
				mult -= x[i + 1] # this A[i, j] is -1
			
			if i - m >= 0:
				mult -= x[i - m] # this A[i, j] is -1
			
			if i + m < dim:
				mult -= x[i + m] # this A[i, j] is -1
			
			# update x[i]
			x[i] = (1 - omega) * x[i] + omega / 4 * (b[0, i] - mult) # A[i, i] = 4 for all i
		
		# update error
		result = mult_mat(A, x)
		error = norm(result - b)
		
		# update iterations
		n_iter += 1
		
		# update graph data
		ite.append(n_iter)
		dif.append(error)
		
	return x, error, n_iter, ite, dif

# finding optimal omega
@njit(fastmath = True, cache = True)
def omega_opt(m):
	return 2 / (1 + math.sin(np.pi / (m + 1)))

def tests(ms, omegas, tol, max_iter = 1e5, plot = True):
	# statistics of each test
	n_iters = []
	errors = []
	solutions = []
	x0s = []
	
	# tests count
	count = 1
	
	# define graph colors
	colors = ['b', 'g', 'r']
	
	for i in range(len(ms)):
		# define m
		m = ms[i]
		
		# add one line in statistics
		n_iters.append([m])
		errors.append([m])
		solutions.append([m])
		x0s.append([m])
		
		# define matrix A and vector b with dim = m²
		A = create_A(m)
		b = create_b(m)
		
		# create x0
		x0 = create_x0(m)
		
		for j in range(len(omegas)):
			# define omega
			omega = omegas[j]
			
			# 0 < omega < 2, so when omega > 2 I consider it with
			# optimal omega
			if omega >= 2:
				omega_in_use = omega_opt(m)
			else:
				omega_in_use = omega
				
			# print actual test values
			print(count, m, omega_in_use)
			
			# copy x0 to avoid overwriting
			x0_j = x0.copy()
			
			# apply SOR
			t = time()
			x, error, n_iter, ite, dif = smart_sor(A, b, omega_in_use, x0_j, tol, max_iter = max_iter)
			tf = time()
			print(f'SOR method with m = {m} and ω = {omega_in_use:.4f} has been complete in {tf - t:.4f} seconds with {n_iter} iterations.\n')
			
			# statistics
			n_iters[i].append(n_iter)
			errors[i].append(error)
			solutions[i].append(x)
			x0s[i].append(x0_j)
			
			# graph
			if plot:
				plt.plot(ite,
						 dif,
						 label = f'ω = {omega_in_use:.4f}',
						 color = colors[j],
						 linewidth = 1)
				plt.axvline(len(ite),
				            color = colors[j],
							linestyle = 'dashed',
							linewidth = 1,
							label = f'End of iterations for ω = {omega_in_use:.4f}')
			
			count += 1
		
		if plot:
			plt.axhline(1e-6,
						color = 'black',
						linestyle = '-',
						linewidth = 1)
			plt.rcParams['figure.figsize'] = (20, 16)
			plt.yscale('log')
			plt.xscale('log')
			plt.xlabel('Iteration')
			plt.ylabel('Error')
			plt.savefig(f'SOR method with m = {m}.png')
			plt.legend('lower left')
			plt.savefig(f'SOR method with m = {m} and legend.png')
			plt.clf()
		
	# print statistics
	header1 = ['m / ω', 'ω'] + omegas[1:]
	header2 = ['m / $omega$', '$omega^*$'] + omegas[1:]
	print('\n\nIterations:')
	print(tabulate(n_iters,
				   headers = header1,
				   tablefmt = 'fancy_grid'))

	print('\nLaTeX code:')
	print(tabulate(n_iters,
				   headers = header2,
				   tablefmt = 'latex'))

	print('\nErrors:')
	print(tabulate(errors,
				   headers = header1,
				   tablefmt = 'fancy_grid'))

	print('\nLaTeX code:')
	print(tabulate(errors,
				   headers = header2,
				   tablefmt = 'latex'))
	
	return n_iters, errors, solutions, x0s

# first call is only to compile functions with numba
tol = 1e-6 # 0.000 001
ms = [5 * i for i in range(1, 3)]
omegas = [10, 1, .5]
n_iters, errors, solutions, x0s = tests(ms, omegas, tol, plot = False)

tol = 1e-6 # 0.000 001
ms = [10, 50, 100, 500, 1000, 5000]
omegas = [10, 1, .5]
max_iter = 1e7
n_iters, errors, solutions, x0s = tests(ms, omegas, tol, max_iter = max_iter)
