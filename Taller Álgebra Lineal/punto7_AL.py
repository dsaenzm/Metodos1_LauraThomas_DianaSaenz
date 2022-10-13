# TALLER 6 - Álgebra Lineal

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time


# 7

s3 = (lambda x,y,z,w: x**2 + y**2 + z**2 + w**2 - 1,)


def GetVectorF(s3,r):    
    dim = len(s3)
    magni = 4
    v = np.zeros(magni)
    
    for i in range(dim):
        v[i] = s3[i](r[0],r[1],r[2],r[3])
        
    return v


def GetJacobian(s3,r,h=1e-6):    
    dim = len(s3)    
    J = np.zeros((dim,4))
    
    for i in range(dim):
        J[i,0] = ( s3[i](r[0]+h,r[1],r[2],r[3]) - s3[i](r[0]-h,r[1],r[2],r[3]) )/(2*h)
        J[i,1] = ( s3[i](r[0],r[1]+h,r[2],r[3]) - s3[i](r[0],r[1]-h,r[2],r[3]) )/(2*h)
        J[i,2] = ( s3[i](r[0],r[1],r[2]+h,r[3]) - s3[i](r[0],r[1],r[2]-h,r[3]) )/(2*h)
        J[i,3] = ( s3[i](r[0],r[1],r[2],r[3]+h) - s3[i](r[0],r[1],r[2],r[3]-h) )/(2*h)
        
    return J.T


def GetMetric(s3,r):
    v = GetVectorF(s3,r)
    return 0.5*np.linalg.norm(v)**2


GetMetric(s3,[1,0,0,0])


def GetFig(F,R,it):    
    fig = plt.figure(figsize=(8,4))
    
    labels = ['X','Y','Z','W']
    
    ax = fig.add_subplot(1,2,1)
    ax1 = fig.add_subplot(1,2,2)

    ax.set_title('Metric: %.20f' %(F[it]))

    ax.plot(F[:it])
    ax.set_xlabel('%.0f' %(it))
    ax.set_yscale('log')
    ax1.plot(R[:it],label=labels)
    ax1.set_xlabel('%.0f' %(it))
    ax1.legend(loc=0)
    
    plt.show()
    
    
def GetSolve(s3,r,lr=1e-3,epochs=int(1e5),error=1e-7):
    d = 1
    it = 0
    Vector_F = np.array([])
    
    R_vector = np.array(r)
    
    while d > error and it < epochs:
        
        CurrentF = GetMetric(s3,r)
        
        J = GetJacobian(s3,r)
        
        s3Vector = GetVectorF(s3,r)
        
        #Machine Learning
        r -= lr*np.dot(s3Vector,J)
        
        R_vector = np.vstack((R_vector,r))
        
        NewF = GetMetric(s3,r)
        
        
        Vector_F = np.append(Vector_F,NewF)
        
        d = np.abs( CurrentF - NewF )/NewF
        
        
        if it%500 == 0:
            #print(it,d)
            clear_output(wait=True)
            GetFig(Vector_F,R_vector,it)
            time.sleep(0.01)
            
        it += 1
        
    if d < error:
        print(' Entrenamiento completo ', d, 'iteraciones', it)
        
    if it == epochs:
        print(' Entrenamiento no completado ')
        
    return r,it,Vector_F,R_vector



r = np.random.uniform(0,1,4)

xsol,it,F,R = GetSolve(s3,r,lr=1e-4)