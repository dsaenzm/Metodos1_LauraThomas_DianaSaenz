# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 14:59:14 2022
"""

# TALLER 5 - MONTE CARLO

import numpy as np
import sympy as sym
import matplotlib.pyplot as plt



# 8)


def T(n):
    return np.math.factorial(n-1)


def f(x,a=2,b=4):
    return ( T(a+b)/( T(a)*T(b) ) ) * (x**(a-1)) * (1-x)**(b-1)



# Cálculo del punto máximo de la función:
    
x = sym.Symbol('x', Real=True)
deriv = sym.diff(f(x), x)

criticos = sym.solve(deriv, x)


maximos = []

for i in criticos:
    if i!=0 and i!=1:
        maximos.append(i)


maximo_x = max(maximos)
maximo = f(maximo_x)



# Distribución - lanzamiento de puntos:

total_lanzados = 1000
cant_adentro = 0
adentro = np.zeros((total_lanzados,2))


for i in range(total_lanzados):
    x = np.random.uniform(0,1)
    y = np.random.uniform(0,float(maximo))
    
    if y < f(x) and x not in adentro:
        adentro[i] = [x,y]
        cant_adentro += 1



# Cálculo del área bajo la curva:

area_total = 1*maximo
integral = (cant_adentro/total_lanzados) * area_total

print('Área bajo la curva:', integral)



# Gráfica:

X = np.linspace(0,1,50)
Y = np.zeros(0)
for i in X:
    Yi = f(i)
    Y = np.append(Y,Yi)


fig = plt.plot(X,Y)

plt.xlabel('x')
plt.ylabel('f (x,2,4)')
plt.title('Función f y puntos caídos bajo la curva')

plt.plot(adentro[:,0], adentro[:,1], 'o')
plt.show()
