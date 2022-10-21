import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from matplotlib import rc
from mpl_toolkits.mplot3d import axes3d

#Punto a
def Temperatura(p):
    x = sym.Symbol('x')
    y = sym.Symbol('y')
    
    suma = 0
    puntos = np.array([[1,1],[-1,1],[-1,-1],[1,-1]])
    for i in range(len(puntos)):
        for j in range(len(puntos[i])):
            
            suma += p[i][j]*(x**i)*(y**j)
    
    return suma

    
#punto b
position = np.zeros((4,2))

for i in range(len(position)):
    for j in range(len(position[i])):
        if i == 0:
            if j == 0:
                position[i][j] = 1
            if j == 1:
                position[i][j] = 1
        if i == 1:
            if j == 0:
                position[i][j] = -1
            if j == 1:
                position[i][j] = 1 
        if i == 2:
            if j == 0:
                position[i][j] = -1
            if j == 1:
                position[i][j] = -1 
        if i == 3:
            if j == 0:
                position[i][j] = 1
            if j == 1:
                position[i][j] = -1

#Punto c

M = np.matrix([[1,1],[-1,1],[-1,-1],[1,-1]])
b = np.matrix([[1], [2] , [0.5] , [0.3]])
x = (M**-1)*b

print(x)

#Punto d
sol = Temperatura(x)

#Punto e

p = [[1,1],[-1,1],[-1,-1],[1,-1]]

X = np.zeros(4)
Y = np.zeros(4)
for i in range(len(p)):
    for j in range(len(p[i])):
        X = np.append(X,Temperatura(p[i]))
        Y = np.append(Y,Temperatura(p[j]))

fig = plt.plot(X,Y)

plt.colormaps()
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Mapa de temperatura')
ax = fig.add_subplot(1,1,1,projection='3d')

plt.show()

#Punto f
print(Temperatura([0.,0.5]))


            
            
        
    
