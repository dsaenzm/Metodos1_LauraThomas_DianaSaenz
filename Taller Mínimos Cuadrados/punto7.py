from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
#%matplotlib notebook
import numpy as np
import matplotlib.pyplot as plt
import os.path as path
import wget
from scipy import optimize

#Descargar datos
file = ''
url = 'https://raw.githubusercontent.com/asegura4488/Database/main/MetodosComputacionalesReforma/Sigmoid.csv'
if not path.exists(file):
    Path_ = wget.download(url,file)
    print('File loaded')
else:
    Path_ = file

data = np.loadtxt('Sigmoid.csv', delimiter=',',skiprows=1)
x = data[:,0]
y = data[:,1]

#Punto A

def Ajuste(x,p0,p1,p2):
    si = p0 / (p1 + np.exp(-p2*x))
    return si

#Punto B

def Fcosto(x,y,a1,a2,a3):
    si = 0
    for i in range(len(x)):
        si += (y[i] - Ajuste(x[i], a1, a2, a3))**2
    return si

#Punto C
h = 0.01
def dx2(x,y,a1,a2,a3):
    
    g = np.zeros(3)
    g[0] = (Fcosto(x,y,a1+h,a2,a3)- Fcosto(x,y,a1-h,a2,a3)) / (2*h) 
    g[1] = (Fcosto(x,y,a1,a2+h,a3)- Fcosto(x,y,a1,a2-h,a3)) / (2*h) 
    g[2] = (Fcosto(x,y,a1,a2,a3+h)- Fcosto(x,y,a1,a2,a3-h)) / (2*h)
    
    return g

#Punto d

p1 = np.array([1,1,1])

def desgrad(p1, itmax, error, lr):
    it = 0
    dist = 1
    while dist > error and it < itmax:
        vgrad = dx2(x,y,p1[0],p1[1],p1[2])
        p1 = p1 - lr*(vgrad)
        dist = np.linalg.norm(p1-optimize.curve_fit(Ajuste,x,y)[0])
        it += 1
        
    return p1,dist,it

#Punto e

error = 0.01
itmax = 20000
lr = 1e-3

a = desgrad(p1, itmax, error, lr)

#Punto f

m1 = a[0][0]
m2 = a[0][1]
m3 = a[0][2]

xx = np.linspace(min(x),max(x),500)
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(1,1,1)
ax.scatter(x,y,c='k',label='Scatter Plot')
plt.plot(xx,Ajuste(xx,m1,m2,m3),c='tab:pink',label='Sigmoid Function')

print("Los parámetros son: \n", m1,m2,m3)
