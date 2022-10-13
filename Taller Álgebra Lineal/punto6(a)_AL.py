# TALLER 6 - Álgebra Lineal

import numpy as np
import matplotlib.pyplot as plt


# 6a

#Sistema:

G=(lambda x,y: np.log(x**2 + y**2) - np.sin(x*y) - np.log(2) - np.log(np.pi), \
   lambda x,y: np.exp(x-y) + np.cos(x*y) )

    
for i in range(2):
    print(G[i](2,2))
print("")


def GetVectorF(G,r):
    dim = len(G)
    v = np.zeros(dim)
    
    for i in range(dim):
        v[i] = G[i](r[0],r[1])
        
    return v

print(GetVectorF(G,[2,2]))


def GetJacobian(G,r,h=1e-6):
    dim = len(G)
    J = np.zeros((dim,dim))
    
    for i in range(dim):
        J[i,0] = ( G[i](r[0]+h,r[1]) - G[i](r[0]-h, r[1]) )/(2*h)
        J[i,1] = ( G[i](r[0],r[1]+h) - G[i](r[0], r[1]-h) )/(2*h)
        
    return J.T

print(GetJacobian(G,[0,1]))


def NewtonRaphson(G,r,error=1e-10):
    it = 0
    d = 1
    Vector_d = np.array([])
    
    J = GetJacobian(G,r)
    while d > error and np.linalg.det(J)!=0:
        it += 1
        
        rc = r
        
        F = GetVectorF(G,r)
        J = GetJacobian(G,r)
        InvJ = np.linalg.inv(J)
        
        r = rc - np.dot( InvJ, F )
        
        diff = r - rc
        print(diff)
        
        d = np.linalg.norm(diff)
        
        Vector_d = np.append( Vector_d , d )
        
    return r,it,Vector_d

r,it,distancias = NewtonRaphson(G,[1,2])
print('\n', r,it)

print('\n', np.round(GetVectorF(G,r)))


plt.plot(distancias)
