import numpy as np

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
	error = norm(A @ x - b)
	n_iter = 0
	while error > tol and n_iter < max_iter:
		for i in range(A.shape[0]):
			mult = 0
			for j in range(A.shape[1]):
				if j != i:
					mult += A[i, j] * x[j]
			
			x[i] = (1 - omega) * x[i] + omega / A[i, i] * (b[i] - mult)
		
		error = norm(A @ x - b)
		n_iter += 1
		
	return x, error, n_iter
