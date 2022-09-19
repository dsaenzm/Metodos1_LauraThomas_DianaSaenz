# TALLER 4 - Integración

import numpy as np
import sympy as sym
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate


# 5)

def g(x):
    return np.exp(-x**2)


class Integrator:
    
    def __init__(self, x,y):        
        self.x = x
        self.y = y
        self.h = self.x[1] - self.x[0]
        
        self.integral = 0.
        
        
    def Trapezoid(self):        
        self.integral = 0.
        self.integral += 0.5*(self.y[0] + self.y[-1])        
        self.integral += np.sum( self.y[1:-1] )
        
        return self.integral*self.h
    
    
    def GetTrapezoidError(self,f):        
        d = (f( self.x + self.h ) - 2*f(self.x) + f( self.x - self.h))/self.h**2                
        max_ = np.max(np.abs(d))
        
        self.error = (max_* (self.x[-1]-self.x[0])**3 )/(12*(len(self.x)-1)**2)        
        return self.error


n = 7
x = np.linspace(0,1,n)
y = g(x)

int1 = Integrator(x,y)
print("Valor integral: ", int1.Trapezoid())
print("Error: ", int1.GetTrapezoidError(g))


# Ejercicios 12-16: cuadratura Gauss-Legendre

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


def GetAllRoots(f,df,x,tolerancia=9):
    roots = np.array([])
    
    for i in x:
        root = NewtonMethod(f, df, i)
        
        if root != False:
            croot = np.round(root,tolerancia)
            
            if croot not in roots:
                roots = np.append(roots,croot)
                
    roots.sort()
    return roots


def GetLegendre(n):
    x = sym.Symbol('x',Real=True)
    y = sym.Symbol('y',Real=True)
    
    y = (x**2-1)**n
    
    p = sym.diff(y,x,n)/(2**n * np.math.factorial(n))
    return p


def GetRootsPolynomial(n,xi,poly,dpoly):    
    x = sym.Symbol('x',Real=True)
    
    pn = sym.lambdify([x],poly[n],'numpy')
    dpn = sym.lambdify([x],dpoly[n],'numpy')
    Roots = GetAllRoots(pn,dpn,xi,tolerancia=8)
    
    return Roots


def GetWeights(Roots,Dpoly):    
    Weights = []
    x = sym.Symbol('x',Real=True)
    dpn = sym.lambdify([x],Dpoly[n],'numpy')
    
    for r in Roots:        
        Weights.append(2 / ( (1-r**2) * dpn(r)**2) )
        
    return Weights


Legendre = []
DerLegendre = []

x = sym.Symbol('x',Real=True)
n=20

for i in range(n+1):    
    poly = GetLegendre(i)    
    Legendre.append(poly)
    DerLegendre.append(sym.diff(poly,x,1))


# 12 )

xi = np.linspace(-1,1,100)
n2 = 2
n3 = 3
Roots2 = GetRootsPolynomial(n2,xi,Legendre,DerLegendre)
Roots3 = GetRootsPolynomial(n3,xi,Legendre,DerLegendre)

Weights2 = GetWeights(Roots2,DerLegendre)
Weights3 = GetWeights(Roots3,DerLegendre)


G = lambda x : 1/x**2

a = 1
b = 2
t2 = 0.5*((b-a)*Roots2 + a + b)
t3 = 0.5*((b-a)*Roots3 + a + b)

Integral_n2 = 0.5*(b-a)*np.sum(Weights2 * G(t2))
Integral_n3 = 0.5*(b-a)*np.sum(Weights3 * G(t3))

print("Valor integral con n=2: ", Integral_n2)
print("Valor integral con n=3: ", Integral_n3)


# 16)

Func = lambda x : x**3 / (np.exp(x) - 1)

# a-)

xi = np.linspace(-1,1,100)
n = 3
Roots = GetRootsPolynomial(n,xi,Legendre,DerLegendre)
Weights = GetWeights(Roots,DerLegendre)

a = 1
b = 10
t = 0.5*((b-a)*Roots + a + b)

Integral = 0.5*(b-a)*np.sum(Weights * G(t))
print("Valor integral con n=3: ", Integral)

# b-)

N = list(range(2,11))
Iest = []

for i in N:
    Roots = GetRootsPolynomial(N,xi,Legendre,DerLegendre)
    Weights = GetWeights(Roots,DerLegendre)
    t = 0.5*((b-a)*Roots + a + b)

    integral = 0.5*(b-a)*np.sum(Weights * G(t))
    Iest.append(integral)
    
Iexact = (np.pi)**4 / 15

e = []
for i in N:
    error = Iest[i]/Iexact
    e.append(error)

graph=pd.Series(e, index=N)
graph.plot(xlabel="e(n)", ylabel="n", title="Laguerre quadrature accuracy")
plt.show

