import numpy as np
from tqdm import tqdm

def f(x,y,z,a,b,c,d,e):
    return (2**(-7))*(x+y+z+a+b+c+d+e)**2

def GetPoints(minimo,maximo): 
    x = np.random.uniform(minimo,maximo)
    y = np.random.uniform(minimo,maximo)
    z = np.random.uniform(minimo,maximo)
    a = np.random.uniform(minimo,maximo)
    b = np.random.uniform(minimo,maximo)
    c = np.random.uniform(minimo,maximo)
    d = np.random.uniform(minimo,maximo)
    e = np.random.uniform(minimo,maximo)
    return x,y,z,a,b,c,d,e

N = int(1e6/2)
Sample = np.zeros(N)
 
for i in tqdm(range(N)):
    x,y,z,a,b,c,d,e = GetPoints(0,1)
    Sample[i] = f(x,y,z,a,b,c,d,e)
        
Integral = np.average(Sample)
print("El valor obtenido es:",str(round(Integral,4)))
print("El valor en la teoría es:",str(round(25/192,4)))
print("Diferencia entre valores: ",np.abs(round(Integral - (25/192),3)))