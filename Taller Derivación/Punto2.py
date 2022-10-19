# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 11:49:43 2022

@author: Lala
"""
# TALLER 2

# Librerías

import numpy as np
import matplotlib.pyplot as plt


# 2) Problema de aplicación


# a. Discretización de ejes X y Y

N = 25

x = np.linspace(-4,4,N)
y = x.copy()

R = 2


# b. Definición de la función de potencial de flujo f

def f(x,y,R=R,V=2):
    potencial = V*x*(1-(R**2)/((x**2)+(y**2)))
    return potencial


# c. Campo de velocidades usando las derivas parciales centrales

def vx(x,y,h=0.001):
    ans = (f(x+h, y) - f(x-h, y)) / 2*h
    return ans

def vy(x,y,h=0.001):
    ans = (f(x, y+h) - f(x, y-h)) / 2*h
    return ans


def campo_v(x,y):
    
    Vx = np.zeros((N,N))
    Vy = Vx.copy()
    
    for i in range(N):
        for j in range(N):
            Vx[i,j] = vx(x[i],y[j])
            Vy[i,j] = - vy(x[i],y[j])
            
    return Vx,Vy

Vx,Vy = campo_v(x,y)


# d. Gráfica del campo de velocidades

fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(1,1,1)

for i in range(N):
    for j in range(N):
        if not x[i]**2+y[j]**2 < R**2:
            ax.quiver(x[i],y[j],Vx[i,j],Vy[i,j],color='b',alpha=0.6)

circle = plt.Circle((0,0),R,color='r',linewidth=2,fill=False)
ax.add_artist(circle)

