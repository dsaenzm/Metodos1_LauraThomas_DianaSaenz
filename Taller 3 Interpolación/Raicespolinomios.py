# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 09:05:37 2022

@author: Lala
"""
# TALLER 3

import numpy as np
import sympy as sym


# Ejercicios raíces

# 3)

def f(x):
    return 3*x**5 + 5*x**4 - x**3


def deriv(f,x,h=1e-6):
    return (f(x+h)-f(x-h))/(2*h)


def NewtonMethod(f,df,xn,itmax=1000,precision=1e-5):    
    error = 1
    it = 0
    
    while error > precision and it < itmax:
        try:
            xn1 = xn - f(xn)/df(f,xn)
            error = np.abs(f(xn)/df(f,xn))
            
        except ZeroDivisionError:
            print('Division entre cero')
            
        xn = xn1
        it += 1
    
    if it == itmax:
        return False
    else:
        return xn


def GetAllRoots(x,tolerancia=6):
    roots = np.array([])
    
    for i in x:
        root = NewtonMethod(f, deriv, i)
        
        if root != False:
            croot = np.round(root,tolerancia)
            
            if croot not in roots:
                roots = np.append(roots,croot)
                
    roots.sort()
    return roots


xtrial = np.linspace(-1000,1000,1000)
roots = GetAllRoots(xtrial)
print("raíces:", roots, "\n")



# 4)

def GetLegendre(n):
    x = sym.Symbol('x',Real=True)
    y = sym.Symbol('y',Real=True)
    
    y = (x**2-1)**n
    
    p = sym.diff(y,x,n)/(2**n * np.math.factorial(n))
    return p


n=20
Legendre = []

for i in range(n):
    Legendre.append(GetLegendre(i))

def GetAllRootsLegendre(X,tolerancia=6):
        roots = np.array([])
        
        for i in X:
            for i in range(n):
                root = NewtonMethod(Legendre[n], deriv, i)
            
            if root != False:
                croot = np.round(root,tolerancia)
                
                if croot not in roots:
                    roots = np.append(roots,croot)
                    
        roots.sort()
        return roots


roots_Legendre = []
for i in range(n):
    root_Legendre = GetAllRootsLegendre(xtrial)
    roots_Legendre.append(root_Legendre)



# 5)

def GetLaguerre(n):
    x = sym.Symbol('x',Real=True)
    y = sym.Symbol('y',Real=True)
    
    y = sym.exp(-x)*x**n
    
    p = sym.exp(x)*sym.diff(y,x,n)/(np.math.factorial(n))
    return p


Laguerre = []

for i in range(n):
    Laguerre.append(GetLaguerre(i))

def GetAllRootsLaguerre(X,tolerancia=6):
        roots = np.array([])
        
        for i in X:
            for i in range(n):
                root = NewtonMethod(Laguerre[n], deriv, i)
            
            if root != False:
                croot = np.round(root,tolerancia)
                
                if croot not in roots:
                    roots = np.append(roots,croot)
                    
        roots.sort()
        return roots


roots_Laguerre = []
for i in range(n):
    root_Laguerre = GetAllRootsLaguerre(xtrial)
    roots_Laguerre.append(root_Laguerre)

