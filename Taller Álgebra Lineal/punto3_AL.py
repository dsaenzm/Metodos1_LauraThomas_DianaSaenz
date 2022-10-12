import numpy as np

A = np.array([[1,0,0],[5,1,0],[-2,3,1]])
B = np.array([[4,-2,1],[0,3,7],[0,0,2]])

Teórico = np.dot(A,B)
print("El resultado teórico es: :\n" , Teórico)

C = np.zeros((A.shape[0],B.shape[1]),dtype = int)

for row in range(3): 
    for col in range(3):
        for elt in range(len(B)):
            C[row, col] += A[row, elt] * B[elt, col]
              
print("El resultado obtenido es: :\n",C)
