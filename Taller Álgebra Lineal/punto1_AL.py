import numpy as np

M = np.array([[3,-1,-1],[-1.,3.,1.],[2,1,4]])
b = np.array([1.,3.,7.])

X0  = np.array([0.,0.,0.])

error = 1e-10
itmax = 1000

# Gauss-Seidel
tamano = np.shape(M)
n = tamano[0]
m = tamano[1]

#  valores iniciales
X = np.copy(X0)
diferencia = np.ones(n, dtype=float)
residuo = 2*error

it = 0
while not(residuo<=error or it>itmax):
    # por fila
    for i in range(0,n,1):
        # por columna
        suma = 0 
        for j in range(0,m,1):
            # excepto diagonal de A
            if (i!=j): 
                suma = suma-M[i,j]*X[j]
        
        nuevo = (b[i]+suma)/M[i,i]
        diferencia[i] = np.abs(nuevo-X[i])
        X[i] = nuevo
    residuo = np.max(diferencia)
    it = it + 1

print("Tomadas los siguientes datos de la clase donde M es: \n", M , "\n y b es: \n", b)
print('La solución del sistema con el método de Gauss_Seidel es: ')
print(X)
print("Llegó a la solución en ", it ," iteraciones")
print("\n Más rápido que el método de Jacobi")