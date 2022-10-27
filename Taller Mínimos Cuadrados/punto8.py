import numpy as np

#PuntoA

A = np.array([[3,1,-1],[1,2,0],[0,1,2],[1,1,-1]])
b = np.array([[-3],[-3],[8],[9]])

ATA = A.T@A
ATb = A.T@b

x = np.linalg.solve(ATA,ATb)

P = A@x

print("Usando mínimos cuadrados matriciales, la proyección es: \n", np.array(P).T[0])


# Punto B

v1 = np.array([3,1,0,1])
v2 = np.array([1,2,1,1])
v3 = np.array([-1,0,2,-1])
B = np.array([-3,-3,8,9])

u1 = v1
u2 = v2 - ((np.dot(v2,u1)/(np.dot(u1,u1)))*u1)
u3 = v3 - ((np.dot(v3,u1)/(np.dot(u1,u1)))*u1) - ((np.dot(v3,u2)/(np.dot(u2,u2)))*u2)

nu1 = u1 / np.linalg.norm(u1)
nu2 = u2 / np.linalg.norm(u2)
nu3 = u3 / np.linalg.norm(u3)

pw = (np.dot(B,nu1)*nu1 + np.dot(B,nu2)*nu2 + np.dot(B,nu3)*nu3)

print("Con el método de Grand-Schmidt, la proyección es: \n" , pw)
print("Con cualquier método da la misma :)")