import numpy as np
import matplotlib.pyplot as plt

def recorrencia(n, a0, a1):
	if n == 0:
		return a0
	elif n == 1:
		return a1
	else:
		a = [a0, a1]
		for i in range(2, n + 1):
			a0, a1 = a1, 22/7 * a1 - 3/7 * a0
			a.append(a1)
			
		return np.array(a)
		
def sequencia(n, a0 = 1, a1 = 1/7):
	a = recorrencia(n, a0, a1)
	return a[-1]

def plot_graph(n, a0 = 1, a1 = 1/7, expression = '(1/7)**i', save_name = 'graph.png'):
	a_real = np.array([eval(expression) for i in range(n + 1)])
	a_alg = recorrencia(n, a0, a1)
	n = np.array([*range(n + 1)])
	
	plt.clf()
	plt.plot(n, a_real, 'b*', label = '1/7^n')
	plt.plot(n, a_alg, 'r*', label = 'Recorrência')
	plt.xlabel('n')
	plt.ylabel('a_n')
	plt.legend()
	plt.savefig(save_name)
	
for n in range(10, 110, 10):
	plot_graph(n, save_name = f'graph with n = {n}.png')
