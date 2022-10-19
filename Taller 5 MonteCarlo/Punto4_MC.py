# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 14:59:14 2022
"""

# TALLER 5 - MONTE CARLO

import numpy as np
import matplotlib.pyplot as plt



# 4)


def C(k=30, N=10**4):
    x = np.random.uniform(0, 1, N)
    
    C = np.zeros(0)
    K = np.linspace(1, k, k)
    
    for k in K:
        suma = 0.
        
        for i in range(1, N-int(k)):
            suma += x[i]*x[i+int(k)]
        suma /= N
        
        C = np.append(C, suma)
          
    return C



# Gráfica:

K = np.linspace(1, 30, 30)
Y = C()

fig = plt.plot(K,Y)

plt.ylim(0.2,0.3)
plt.xlabel('k-ésimo vecino')
plt.ylabel('C(k)')
plt.title('Correlación de los primeros k=30 vecinos, en función de de k')
plt.show()
