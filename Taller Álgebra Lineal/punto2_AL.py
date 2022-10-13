import numpy as np

class MetodosSistemasLineales:
    
    def __init__(self,A,B):
        self.A = A
        self.B = B
        self.itmax = 1000
        self.error = 1e-10
    
    def Metodo1(self):
        
        M,N = self.A.shape
        x = np.zeros(N)
        sumk = np.zeros_like(x)
        x = [13,20,-1]
        it = 0
        residuo = np.linalg.norm( self.B - np.dot(self.A,x) )
        
        while ( residuo > self.error and it < self.itmax ):
            it += 1
            for i in range(M):
                sum_ = 0
                for j in range(N):
                    if i != j:
                        sum_ += self.A[i][j]*x[j]
                sumk[i] = sum_
            
            for i in range(M):
                if self.A[i,i] != 0:
                    x[i] = (self.B[i] - sumk[i])/self.A[i,i]
                else:
                    print('No invertible con el método 1 (Jacobi)')
        

            residuo = np.linalg.norm( self.B - np.dot(self.A,x) )
        
        return "Con el primer método, la solución es:",x, " con ",it," iteraciones "
    
    def Metodo2(self):
        
        X0  = np.array([0.,0.,0.])
        tamano = np.shape(self.A)
        n = tamano[0]
        m = tamano[1]
        X = np.copy(X0)
        diferencia = np.ones(n, dtype=float)
        it = 0
        residuo=2*self.error
        while not(residuo<=self.error or it>self.itmax):
            for i in range(0,n,1):
                suma = 0 
                for j in range(0,m,1):
                    if (i!=j): 
                        suma = suma- self.A[i,j]*X[j]
                nuevo = (self.B[i]+suma)/ self.A[i,i]
                diferencia[i] = np.abs(nuevo-X[i])
                X[i] = nuevo
            residuo = np.max(diferencia)
            it = it + 1
        
        if (it > self.itmax):
            print("No se puede resolver con el método 2 (Gauss-Seidel)")
        return "Con el segundo método la solución es ", X , " con ", it, " iteraciones "

#Probar

M = np.array([[3,-1,-1],[-1.,3.,1.],[2,1,4]])
b = np.array([1.,3.,7.])

print("Para comprobar, se usan los datos de la matriz como: \n", M , "\n y el vector es: \n", b)    
a = MetodosSistemasLineales(M, b)
print("\n El primer método es el de Jacobi y el segundo el de Gauss-Seidel \n")
print(a.Metodo1())
print(a.Metodo2())
