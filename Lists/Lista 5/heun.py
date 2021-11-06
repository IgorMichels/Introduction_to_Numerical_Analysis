import numpy as np
import matplotlib.pyplot as plt

#      x'' + x * sin(t) - x' * cos(t) = 0
# <==> x'' = x' * cos(t) - x * sin(t)
# defina f(t, x, x') = x' * cos(t) - x * sin(t)
# logo, x'' = f(t, x, x').
# Defina y = x', assim, y' = f(t, x, y), ou seja
# x'' = x' * cos(t) - x * sin(t)
# y' = y * cos(t) - x * sin(t)

def f1(t, x, y):
	return y

def f2(t, x, y):
	return y * np.cos(t) - x * np.sin(t)
	
def heun(x0, y0, h, tmin = 0, tmax = 4):
	t = np.arange(tmin, tmax + h, h)
	x, y = np.zeros(t.shape), np.zeros(t.shape)
	x[0] = x0
	y[0] = y0
	
	for i in range(1, len(t)):
		k1 = f1(t[i - 1], x[i - 1], y[i - 1])
		m1 = f2(t[i - 1], x[i - 1], y[i - 1])
		
		k2 = f1(t[i], x[i - 1] + k1 * h, y[i - 1] + m1 * h)
		m2 = f2(t[i], x[i - 1] + k1 * h, y[i - 1] + m1 * h)
		
		x[i] = x[i - 1] + h * np.mean([k1, k2])
		y[i] = y[i - 1] + h * np.mean([m1, m2])
	
	return t, x
	
t, x = heun(1, 1, 1/2)
plt.plot(t, x, 'red')

t, x = heun(1, 1, 1/8)
plt.plot(t, x, 'blue')

t, x = heun(1, 1, 1/16)
plt.plot(t, x, 'green')

t = np.arange(0, 4 + 1/16, 1/16)
x = np.exp(np.sin(t))
plt.plot(t, x, 'black')

plt.show()
