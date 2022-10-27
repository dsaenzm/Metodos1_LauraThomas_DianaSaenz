# TALLER 7 - Mí­nimos cuadrados

import numpy as np
import sympy as sym
import matplotlib.pyplot as plt


# 1)

# a-

y1 = lambda x: 2*x - 2
y2 = lambda x: 0.55 - x/2
y3 = lambda x: 4 - x


def GetFit(x,y,n):
    DataSize = x.shape[0]
    b = y
    A = np.ones((DataSize,n+1))
    
    for i in range(1,n+1):
        A[:,i] = x**i
    
    AT = np.dot(A.T,A)
    bT = A.T @ b

    xsol = np.linalg.solve(AT,bT)
    return xsol

x = np.arange(0,8,0.2)
y = y1(x)
y += y2(x)
y += y3(x)

param = GetFit(x,y,1)


def GetModel(x,p):
    y = 0
    for n in range(len(p)):
        y += p[n]*x**n
        
    return y

X = sym.Symbol('x',real=True)
X_ = param[0]
X = -1.8*X_
Y_ = GetModel(X,param)
print((X_,Y_))


x_ = range(0,10)
plt.ylim(-4,4)
plt.plot(x_, [y1(i) for i in x_])
plt.plot(x_, [y2(i) for i in x_])
plt.plot(x_, [y3(i) for i in x_])
plt.plot(X_, Y_, 'o')
plt.show


# b-

dy1 = lambda x: 2
dy2 = lambda x: 1/2
dy3 = lambda x: -1


h = 0.03
x__ = np.arange(-5,5,1)
y__ = np.arange(-5,5,h)
pt = np.zeros((3))
for i in x__:
    for j in y__:
        np.append(pt[0],y1(i))
        np.append(pt[1],y2(i))
        np.append(pt[2],y3(i))
    
#d = min(pt.sum())
#print(d)
