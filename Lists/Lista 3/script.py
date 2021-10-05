import numpy as np
from scipy.optimize import minimize

def f(x):
	return np.log(15 - np.log(x)) - x
	
def f_prime(x):
	return - (1 / (x*(15 - np.log(x)))) - 1
	
def f_prime_abs(x):
	return abs(f_prime(x))
	
def f_prime_prime(x):
	return (14 - np.log(x)) / (x**2 * (15 - np.log(x))**2)
	
def f_prime_prime_abs(x):
	return - abs(f_prime_prime(x))
	
def g(x):
	return np.log(15 - np.log(x))
	
def g_prime(x):
	return - 1 / (x * (15 - np.log(x)))
	
def g_prime_abs(x):
	return - abs(- 1 / (x * (15 - np.log(x))))
	
def fix_point(x0, tol):
	x = x0
	k = minimize(g_prime_abs, 2, bounds = [(1, 3)])
	k = - k.fun
	dif = tol + 1
	i = 0
	while dif > tol:
		x0 = x
		x = g(x0)
		dif = k / (1 - k) * abs(x - x0)
		i += 1
		
	return x, i
	
def newton_raphson(x0, tol):
	x = x0
	M = minimize(f_prime_prime_abs, 2, bounds = [(1, 3)])
	m = minimize(f_prime_abs, 2, bounds = [(1, 3)])
	M = - M.fun
	m = m.fun
	dif = tol + 1
	i = 0
	while dif > tol:
		x0 = x
		x = x0 - f(x0)/f_prime(x0)
		dif = M / (2 * m) * abs(x - x0) ** 2
		i += 1
		
	return x, i
	
x0 = float(input('Qual o x0 para o método iterativo que visa encontrar x^* com 5 algarismos significativos corretos? '))
x, i = fix_point(x0, 1e-5)
print(f'x^* com 5 algarismos significativos é {x} e foi encontrado com {i} iterações')

print()
x, i = newton_raphson(2, 1e-8)
print(f'Agora, usando x0 = 2 e visando 8 algarismos significativos corretos, com o método de Newton-Raphson, temos x^* = {x}, encontrado com {i} iterações.')
