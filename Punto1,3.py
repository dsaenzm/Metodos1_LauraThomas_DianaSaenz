#Punto 3

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-10,10,0.05)
h = 0.05
f = (1 + np.exp(-(x**2)))**(-0.5)

def DerivadaCentral (f,h):
    M = [1,0,-1]
    df = np.zeros(400)
    
    for n in range(1,400):
        sumatoria = 0
        
        for m in range(-1,1):

            sumatoria = sumatoria + (M[m] * f[n + m])
            df[n] = sumatoria/ (2*h)
            
 
    
    return df



#Graficar la derivada(Cómo se comporta la función)
df = (DerivadaCentral(f, h))
plt.plot(x,df)

#Punto 4
#A
print("El kernel de convolución será [1,-2,1]")
#Esto es por la secuencia que hace la expresión matemática. En la primera, los coeficientes principales en el numerador 
#dan el kernel, así como en la segunda derivada, también lo darán dichos coeficientes

def SegundaDerivada (f,h):
    M = [1,-2,1]
    df = np.zeros(400)
    
    for n in range(1,400):
        sumatoria = 0
        
        for m in range(-1,1):

            sumatoria = sumatoria + (M[m] *f[n + m])
            df[n] = sumatoria/ (h**2)
            
 
    
    return df

#df1 = (SegundaDerivada(f, h))
#plt.plot(x,df1)




