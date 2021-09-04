import numpy as np

def sor(A, b, omega, x0, tol, norm = np.inf, max_iter = 10000):
	'''
	solves Ax = b system with SOR method with w = omega
	and tolerance tol in the given norm 
	'''
	
	if norm == np.inf:
		norm = lambda x : np.linalg.norm(x, np.inf)
	elif type(norm) == int:
		norm = lambda x : np.linalg.norm(x, norm)
	
	x = x0
	error = norm(A @ x - b)
	n_iter = 0
	while error > tol and n_iter < max_iter:
		for i in range(A.shape[0]):
			sigma = 0
			for j in range(A.shape[1]):
				if j != i:
					sigma += A[i, j] * x0[j]
			
			x[i] = (1 - omega) * x[i] + omega / A[i, i] * (b[i] - sigma)
		
		error = norm(A @ x - b)
		n_iter += 1
		
	return x, error, n_iter
