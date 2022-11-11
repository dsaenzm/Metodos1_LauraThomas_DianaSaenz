# TALLER 8 - Probabilidad

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Generales de probabilidad


# 9) Superficie de probabilidad

x = np.linspace(0.1, 0.9, 1000)
y = np.linspace(0.1, 0.5, 1000)

X,Y = np.meshgrid(x,y)
Z = (X+Y-2*X*Y+1)/4

Max = np.amax(Z)
Min = np.amin(Z)

print('Probabilidad máxima: ',round(Max,3))
print('Probabilidad mínima: ',round(Min,3))


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, Z, cmap="inferno")

