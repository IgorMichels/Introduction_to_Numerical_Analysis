#      x'' + x * sin(t) - x' * cos(t) = 0
# <==> x'' = x' * cos(t) - x * sin(t)
# defina f(t, x, x') = x' * cos(t) - x * sin(t)
# logo, x'' = f(t, x, x').
# Defina y = x', assim, y' = f(t, x, y), ou seja
# x'' = x' * cos(t) - x * sin(t)
# y' = y * cos(t) - x * sin(t)

import numpy as np

def f(t, x, y):
	return y * np.cos(t) - x * np.sin(t)
	
def heun(x0, y0, h, tmin = 0, tmax = 4):
	t = np.arange(tmin, tmax + h, h)
	x, y = np.zeros(t.shape), np.zeros(t.shape)
	x[0] = x0
	y[0] = y0
	
	for i in range(1, len(t)):
		k1 = y[i - 1]
		m1 = f(t[i - 1], x[i - 1], y[i - 1])
		
		k2 = y[i - 1] + m1 * h
		m2 = f(t[i], x[i - 1] + k1 * h, y[i - 1] + m1 * h)
		
		x[i] = x[i - 1] + h * np.mean([k1, k2])
		y[i] = y[i - 1] + h * np.mean([m1, m2])
	
	return t, x
