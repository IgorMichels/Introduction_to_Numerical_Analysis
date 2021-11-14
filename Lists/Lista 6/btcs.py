import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import solve
from mpl_toolkits import mplot3d

def f(x):
	return np.sin(np.pi / 2 * x) ** 2
	
def g(t):
	return 0
	
def h(t):
	return 0
	
def create_matrix(v, dim, kd):
	A = (1 - kd + 2 * v) * np.eye(dim)
	A[:-1, 1:] -= v * np.eye(dim - 1)
	A[1:, :-1] -= v * np.eye(dim - 1)
	
	return A
	
def btcs(c, d, L, T, dt, dx, f, g, h):
	'''
	calcula a solução u(t, x) para
	o problema dado pelas equações
	u_t = c * u_{xx} + d * u com
	u(x, 0) = f(x), (0 <= x <= L)
	u(0, t) = g(t), t > 0
	u(L, t) = h(t), t > 0
	para os valores (x, t) dentro
	do retângulo [0, L] x [0, T].
	'''
	
	x1 = np.arange(0, L + dx, dx)
	t1 = np.arange(0, T + dt, dt)
	u = f(x1)
	x = np.matrix([i for j in range(len(t1)) for i in x1])
	t = np.matrix([i for i in t1 for j in range(len(x1))])
	x = x.reshape(len(x1), len(t1))
	t = t.reshape((len(t1), len(x1)))
	u = np.matrix(u.reshape(1, u.size))
	
	v = c * dt / (dx ** 2)
	kd = dt * d
	A = create_matrix(v, len(x1) - 2, kd)
	
	for i in range(len(t1) - 1):
		y = u[i, 1:-1].T
		y[0] += v * g(t[i + 1, 0])
		y[-1] += v * g(t[i + 1, -1])
		
		s = solve(A, y)
		s = np.hstack((np.array([0]), np.array(s.T)[0], np.array([0])))
		u = np.vstack((u, s))
		
	u = np.array(u.reshape(1, u.size))[0]
	x = np.array(x.reshape(1, x.size))[0]
	t = np.array(t.reshape(1, t.size))[0]
	
	ax = plt.axes(projection = '3d')
	ax.plot_trisurf(x, t, u, cmap = 'viridis')
	plt.show()
	
c = 1
d = 1
L = 4
T = 10
dt = 0.01
dx = 0.05
btcs(c, d, L, T, dt, dx, f, g, h)
