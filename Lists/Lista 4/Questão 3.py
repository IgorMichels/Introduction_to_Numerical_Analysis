from matplotlib.backend_bases import MouseButton
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np

def lagrange_pol(x, y, x_graph):
	soma = 0
	y_graph = np.zeros(len(x_graph))
	for i in range(len(x)):
		prod = 1
		x_aux = np.ones(len(x_graph))
		for j in range(len(x)):
			if i != j:
				x_aux *= (x_graph - x[j])/(x[i] - x[j])
			
		x_aux *= y[i]
		y_graph += x_aux
		
	return y_graph

def pol(points, old_points, xlim = [-5, 5], ylim = [-5, 5]):
	plt.clf()
	old_points = deepcopy(points)
	x = [point[0] for point in points]
	y = [point[1] for point in points]
	if x != []:
		if xlim[0] + 1 >= min(x):
			xlim[0] = min(x) - 1
			
		if xlim[1] - 1 <= max(x):
			xlim[1] = max(x) + 1
			
		if ylim[0] + 1 >= min(y):
			ylim[0] = min(y) - 1
			
		if ylim[1] - 1 <= max(y):
			ylim[1] = max(y) + 1
			
	if len(x) == 0:
		# sem pontos, sem polinômio
		plt.plot()
	elif len(x) == 1:
		plt.scatter(x, y)
		# existem infinitas retas nesse caso, então
		# tomei a vertical
		plt.hlines(y[0], xlim[0], xlim[1])
		plt.plot()
	else:
		plt.scatter(x, y)
		x_graph = np.arange(xlim[0], xlim[1], 0.01)
		y_graph = lagrange_pol(x, y, x_graph)
		plt.plot(x_graph, y_graph)
	
	plt.xlim(xlim)
	plt.ylim(ylim)
	plt.connect('button_press_event', on_click)
	plt.show()

def on_click(event):
    if event.button is MouseButton.LEFT:
    	if event.xdata not in [point[0] for point in points]:
    		points.append((event.xdata, event.ydata))
    		pol(points, old_points)

points = []
old_points = None
pol(points, old_points)
