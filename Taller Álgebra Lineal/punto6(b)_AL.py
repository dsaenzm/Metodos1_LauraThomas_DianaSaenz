# TALLER 6 - Álgebra Lineal

import numpy as np
import matplotlib.pyplot as plt


# 6a

#Sistema:

G=(lambda x,y: np.log(x**2 + y**2) - np.sin(x*y) - np.log(2) - np.log(np.pi), \
   lambda x,y: np.exp(x-y) + np.cos(x*y) )
    
G=(lambda x,y,z: 6*x - 2*np.cos(y*z) - 1, \
   lambda x,y,z: 4*y + (x**2 + np.sin(z) + 1.06)**0.5 + 0.9, \
   lambda x,y,z: np.exp(-x*y) + 20*z + 9.471975 )

    
for i in range(3):
    print(G[i](0,0,0))
print("")


def GetVectorF(G,r):
    dim = len(G)
    v = np.zeros(dim)
    
    for i in range(dim):
        v[i] = G[i](r[0],r[1],r[2])
        
    return v

print(GetVectorF(G,[0,0,0]),'\n')


def GetJacobian(G,r,h=1e-6):
    dim = len(G)
    J = np.zeros((dim,dim))
    
    for i in range(dim):
        J[i,0] = ( G[i](r[0]+h,r[1],r[2]) - G[i](r[0]-h,r[1],r[2]) )/(2*h)
        J[i,1] = ( G[i](r[0],r[1]+h,r[2]) - G[i](r[0],r[1]-h,r[2]) )/(2*h)
        J[i,2] = ( G[i](r[0],r[1],r[2]+h) - G[i](r[0],r[1],r[2]-h) )/(2*h)
        
    return J.T

print(GetJacobian(G,[0,1,-1]),'\n')


def NewtonRaphson(G,r,error=1e-10):
    it = 0
    d = 1
    Vector_d = np.array([])
    
    while d > error:
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

r,it,distancias = NewtonRaphson(G,[1,2,3])
print('\n', r,it)

print('\n', np.round(GetVectorF(G,r)))


plt.plot(distancias)