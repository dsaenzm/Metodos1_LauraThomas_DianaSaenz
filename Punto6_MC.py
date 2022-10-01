import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def CreateSphere(N,r):   
    X = np.zeros(N)
    Y = np.zeros_like(X)
    Z = np.zeros_like(X)    
    for i in tqdm(range(N)):  
        u = np.random.rand()
        v = np.random.rand()
        angulo = 2*np.pi*u
        p = np.math.acos((2*v)-1)
        X[i] = r*np.cos(angulo)*np.sin(p)
        Y[i] = r*np.sin(p)*np.sin(angulo)
        Z[i] = r*np.cos(p)
    return X,Y,Z

X,Y,Z = CreateSphere(10000,1)
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1,1,1, projection = '3d')
ax.set_xlim3d(-1,1)
ax.set_ylim3d(-1,1)
ax.set_zlim3d(-1,1)
ax.view_init(22,22)
ax.scatter(X,Y,Z, color = "g")

def funcion(x,y,z,angulo):
    return np.sin(angulo)*np.exp(np.sqrt(x**2+y**2+z**2))

def GetPoints(angulo_minimo,angulo_maximo,p_minimo,p_maximo,r): 
    u = np.random.rand()
    r_ = r*u**(1/4)
    angulo = np.random.uniform(angulo_minimo,angulo_maximo)
    p = np.random.uniform(p_minimo,p_maximo)
    x = r_*np.cos(angulo)*np.sin(p)
    y = r_*np.sin(p)*np.sin(angulo)
    z = r_*np.cos(p)
    return x,y,z,angulo

N = int(1e4)
Sample = np.zeros(N)
    
for i in range(N):
    x,y,z,angulo = GetPoints(0,np.pi,0,2*np.pi,1)
    Sample[i] = funcion(x,y,z,angulo)
        
Integral = 2*np.pi*np.average(Sample)
print("El valor obtenido es:", str(round(Integral,12)))
print("El valor en teoría es:", str(round(4*np.pi*(np.e-2),12)))
print("Diferencia entre los valores:",np.abs(round(Integral - (4*np.pi*(np.e-2)),6)))