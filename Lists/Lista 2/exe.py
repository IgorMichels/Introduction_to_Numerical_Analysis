from tabulate import tabulate
from create_system import *
from sor import *
import math

# finding optmal omega
omega_opt = lambda m : 2 / (1 + math.sin(np.pi / (m + 1)))

tol = 10e-6
ms = [50, 100, 1000, 5000]
omegas = [omega_opt, 1, .5]
n_iters = [[ms[i]] + list(np.zeros(len(omegas), int)) for i in range(len(ms))]
errors = [[ms[i]] + list(np.zeros(len(omegas), int)) for i in range(len(ms))]
solutions = [[ms[i]] + list(np.zeros(len(omegas), int)) for i in range(len(ms))]

for i in range(len(ms)):
	m = ms[i]
	A = create_A(m)
	b = create_b(m)
	x0 = np.random.normal(2, .0001, A.shape[0])
	for j in range(len(omegas)):
		omega = omegas[j]
		if omega == omega_opt:
			omega = omega_opt(m)
		
		x0_j = x0.copy()
		x, error, n_iter = sor(A, b, omega, x0_j, tol, np.inf)
		n_iters[i][j + 1] = n_iter
		
print(tabulate(n_iters,
			   headers = ['m\ω', 'ω ótimo', 1, 0.5],
			   tablefmt = 'fancy_grid'))
		
print(tabulate(n_iters,
			   headers = ['m\ω', 'ω ótimo', 1, 0.5],
			   tablefmt = 'latex'))
		
print(tabulate(errors,
			   headers = ['m\ω', 'ω ótimo', 1, 0.5],
			   tablefmt = 'fancy_grid'))
		
print(tabulate(errors,
			   headers = ['m\ω', 'ω ótimo', 1, 0.5],
			   tablefmt = 'latex'))




'''
omega = 1
tol = 10e-6
norm = 'inf'
A = create_A(m)
b = create_b(m)
x0 = np.zeros((A.shape[0], 1))

x, error, n_iter = sor(A, b, omega, x0, tol, np.inf)
print(x)
print(n_iter)

print('ω')
'''
