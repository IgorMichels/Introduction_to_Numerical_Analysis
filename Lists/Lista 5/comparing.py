import matplotlib.pyplot as plt
import numpy as np

from heun import heun
from rk4 import rk4

def comparing(x0, y0, h, tmin = 0, tmax = 4):
	t, x = rk4(x0, y0, h, tmin, tmax)
	plt.plot(t, x, 'red', label = 'RK4')
	
	t, x = heun(x0, y0, h, tmin, tmax)
	plt.plot(t, x, 'blue', label = 'Heun')
	
	t = np.arange(0, 4 + 1/(2 ** 10), 1/(2 ** 10))
	x = np.exp(np.sin(t))
	plt.plot(t, x, 'black', label = 'x(t) = exp(sin(t))')
	
	plt.legend()
	plt.title(f'Comparação RK4, Heun e Solução com h = {h}')
	plt.savefig(f'h = {h}.png')
	plt.show()
	
for i in range(1, 5):
	comparing(1, 1, 1/(2 ** i))
	
# Ao comentar a linha 16 podemos ver que ambos os métodos
# obtém boas aproximações para a função, entretanto o RK4
# é mais preciso desde os primeiros passos
