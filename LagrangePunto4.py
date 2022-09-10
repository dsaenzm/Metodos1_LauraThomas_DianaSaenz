import numpy as np
import pandas as pd
import sympy as sym
import os.path as path
import wget
import math


file = 'Parabolico.csv'
url = 'https://raw.githubusercontent.com/asegura4488/Database/main/MetodosComputacionalesReforma/Parabolico.csv'
if not path.exists(file):
    Path_ = wget.download(url,file)
else:
    Path_ = file

Data = pd.read_csv(Path_,sep=',')

X = np.float64(Data['X'])
Y = np.float64(Data['Y'])
Diff = np.zeros((len(X),len(Y)))
h = 1

def Lagrange(x,xi,j):
    
    prod = 1.0
    n = len(xi)
    
    for i in range(n):
        if i != j:
            prod *= (x - xi[i])/(xi[j]-xi[i])
            
    return prod
def Poly(x,xi,yi):
    
    Sum = 0.
    n = len(xi)
        
    for j in range(n):
        Sum += yi[j]*Lagrange(x,xi,j)
        
    return Sum

x = np.linspace(0,7,100)
y = Poly(x,X,Y)



xsym = sym.Symbol('x')
f = Poly(xsym,X,Y)
f = f.expand()


def derivada(f,x):
  return (f.subs(xsym,x+h)-f.subs(xsym,x-h))/(2*h)



def GetNewtonMethod(f,df,xn,itmax=1000,precision=1e-5):
    error = 1
    it = 0
    while error > precision and it < itmax: 
        try:
            xn1 = xn - f.subs(xsym,xn)/df(f,xn)
            error = np.abs(f.subs(xsym,xn)/df(f,xn))
        except ZeroDivisionError:
            print('Hay un error de división por cero.')     
        xn = xn1
        it += 1
    if it == itmax:
        return False
    else:
        return xn


def GetAllRoots(x,tolerancia=8):
    Roots = np.array([])
    for i in x:
        root = GetNewtonMethod(f,derivada,i)
        if root != False:
            croot = round(root,tolerancia)
            if round(croot,2) not in Roots:
                if np.abs(croot) < 0.001:
                    Roots = np.append(Roots,round(croot,8))
                else:   
                    Roots = np.append(Roots,round(croot,8))
                    
    Roots.sort()
    return Roots


ex = np.linspace(1,100,100)
Raices = GetAllRoots(ex)

Resultado = []
for i in Raices:
    if i not in Resultado:
        Resultado.append(i)

#Segundo termino será la cuadratica, el primero, el lineal en la lista de Resultado.

maximo = max(y)
#La función alcanza la velocidad y en 0 en el maximo término de y, cuando empieza a bajar 
#Esto es útil para (Vy = (Voy**2) + 2g(y-yo)) Donde Vy se toma como 0, yo como 0 y y como maximo

Voy = (2*(9.8)*maximo) ** (1/2)    
#El coeficiente de x va a ser Voy/Vox

x = sym.Symbol('x')
f = Poly(x,X,Y)
f = sym.expand(f)

#print(f)
#Con esto podemos ver el coeficiente que acompaña la x.
#Por lo tanto Vox = Voy/0.363970234266202

Vox = Voy/(0.363970234266202)

#Ahora por pitágoras se puede saber Vo

Vo = (Vox**2+Voy**2) ** (1/2)

#Se redondea para que de lo del ejercicio, sin embargo, da casi exacto
print("Vo es = " , round(Vo,2))


theta_radianes = (math.asin((Resultado[1]*(-9.8))/(Vo**2)))/2
thetha_grados = np.abs((theta_radianes*180)/np.pi)

print("El ángulo es: ", round(thetha_grados,2))