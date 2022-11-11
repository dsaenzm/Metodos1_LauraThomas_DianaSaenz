# TALLER 8 - Probabilidad

import numpy as np
import matplotlib.pyplot as plt


# Generales de probabilidad


# 4)

def ProbCumple(n):
    prob = 1
    N = 365 - n
    
    for i in range(N,365):
        prob *= (i/365)
        
    return 1-prob


x = np.array(range(80+1))
y = np.zeros(0)
for i in x:
    yi = ProbCumple(i)
    y = np.append(y, yi)


plt.plot(x,y)
plt.xlabel('n')
plt.ylabel('P(n)')
plt.title('Probabilidad de n <= 80')

